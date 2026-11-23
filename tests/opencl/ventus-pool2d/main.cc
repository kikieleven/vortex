// pool2d (Ventus-OpenCL-Testcase, _get_case/AIops/Pool2D) — standalone
// self-checking OpenCL port for Vortex.
//
//   y[c,oy,ox] = max or average over the KH x KW window at
//                (oy*SH - PH, ox*SW - PW)
//   mode 0 = max, mode 1 = average; `count_include_pad` selects whether the
//   padded (out-of-bounds) positions count in the average denominator.
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/Pool2D/ —
// the directory has only pool2d.cl, so the host and CPU reference below are
// new. The CPU reference reproduces the kernel path exactly, including the
// `-FLT_MAX` starting accumulator for max mode and the empty-window fallbacks.
//
// Notes:
//   * the launch keeps the upstream 3D NDRange {C, OH, OW}; local {1,4,4},
//     global rounded up, kernel guard drops the extra work-items.
//   * inputs come from a private deterministic LCG (porting guide §13.2).
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL. Max is
//     an exact select, but the average path divides, so a tolerance is used.
//     A window with no valid position yields -FLT_MAX in max mode on both
//     sides, which still compares equal.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "pool2d"

#define REL_TOL 1e-5f
#define ABS_TOL 1e-6f

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     cleanup();                                                        \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       cleanup();                                                      \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

// Deterministic LCG (porting guide §13.2).
static uint32_t lcg_state = 1u;
static float generate_pm1() {                 // [-1, 1)
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<float>(lcg_state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    return -1;
  }

  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

// Same window scan, same max/avg paths, same empty-window fallbacks as kernel.cl.
static void pool2d_cpu(float* y, const float* x,
                       int C, int IH, int IW, int OH, int OW,
                       int KH, int KW, int SH, int SW, int PH, int PW,
                       int mode, int count_include_pad) {
  const int inHW = IH * IW;
  const int outHW = OH * OW;
  for (int c = 0; c < C; ++c) {
    for (int oy = 0; oy < OH; ++oy) {
      for (int ox = 0; ox < OW; ++ox) {
        float acc = (mode == 0) ? (-FLT_MAX) : 0.0f;
        int count = 0;
        for (int ky = 0; ky < KH; ++ky) {
          const int iy = oy * SH - PH + ky;
          for (int kx = 0; kx < KW; ++kx) {
            const int ix = ox * SW - PW + kx;
            const int valid = (iy >= 0 && iy < IH && ix >= 0 && ix < IW);
            if (mode == 0) {
              if (valid) {
                const int idx = c * inHW + iy * IW + ix;
                acc = fmaxf(acc, x[idx]);
              }
            } else {
              float v = 0.0f;
              if (valid) {
                const int idx = c * inHW + iy * IW + ix;
                v = x[idx];
              }
              if (count_include_pad) {
                acc += v;
                ++count;
              } else if (valid) {
                acc += v;
                ++count;
              }
            }
          }
        }
        if (mode == 1) {
          if (count > 0) acc /= (float)count;
          else           acc = 0.0f;
        }
        y[c * outHW + oy * OW + ox] = acc;
      }
    }
  }
}

static size_t round_up(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem x_memobj = NULL;
cl_mem y_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (x_memobj) clReleaseMemObject(x_memobj);
  if (y_memobj) clReleaseMemObject(y_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int dim_c = 2;
int in_size = 16;
int k_size = 3;
int stride = 2;
int pad = 1;
int mode = 0;                 // 0=max, 1=avg
int count_include_pad = 0;

static void show_usage() {
  printf("Usage: [-c C] [-n IH=IW] [-k kernel] [-s stride] [-p pad]\n"
         "       [-a mode(0=max,1=avg)] [-i count_include_pad(0|1)] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "c:n:k:s:p:a:i:h")) != -1) {
    switch (c) {
    case 'c':
      dim_c = atoi(optarg);
      break;
    case 'n':
      in_size = atoi(optarg);
      break;
    case 'k':
      k_size = atoi(optarg);
      break;
    case 's':
      stride = atoi(optarg);
      break;
    case 'p':
      pad = atoi(optarg);
      break;
    case 'a':
      mode = atoi(optarg);
      break;
    case 'i':
      count_include_pad = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }

  printf("float workload C=%d IH=IW=%d kernel=%d stride=%d pad=%d mode=%s count_include_pad=%d\n",
         dim_c, in_size, k_size, stride, pad, mode == 0 ? "max" : "avg",
         count_include_pad);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (dim_c < 1 || in_size < 1 || k_size < 1 || stride < 1 || pad < 0 ||
      mode < 0 || mode > 1 || count_include_pad < 0 || count_include_pad > 1) {
    printf("Error: invalid parameters\n");
    return -1;
  }

  const int C = dim_c;
  const int IH = in_size;
  const int IW = in_size;
  const int KH = k_size;
  const int KW = k_size;
  const int SH = stride;
  const int SW = stride;
  const int PH = pad;
  const int PW = pad;
  const int OH = (IH + 2 * PH - KH) / SH + 1;
  const int OW = (IW + 2 * PW - KW) / SW + 1;
  if (OH <= 0 || OW <= 0) {
    printf("Error: invalid output size OH=%d OW=%d\n", OH, OW);
    return -1;
  }

  const size_t in_n = (size_t)C * IH * IW;
  const size_t out_n = (size_t)C * OH * OW;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, in_n * sizeof(float), NULL, &_err));
  y_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_n * sizeof(float), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int arg = 0;
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&y_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&C));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&IH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&IW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&OH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&OW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&KH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&KW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&SH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&SW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&PH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&PW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&mode));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&count_include_pad));

  std::vector<float> h_x(in_n), h_y(out_n, 0.0f), h_ref(out_n);
  for (size_t i = 0; i < in_n; ++i)
    h_x[i] = generate_pm1();

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, in_n * sizeof(float), h_x.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t local_work_size[3] = {1, 4, 4};
  size_t global_work_size[3];
  global_work_size[0] = round_up((size_t)C, local_work_size[0]);
  global_work_size[1] = round_up((size_t)OH, local_work_size[1]);
  global_work_size[2] = round_up((size_t)OW, local_work_size[2]);
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, y_memobj, CL_TRUE, 0, out_n * sizeof(float), h_y.data(), 0, NULL, NULL));

  printf("Verify result\n");
  pool2d_cpu(h_ref.data(), h_x.data(), C, IH, IW, OH, OW,
             KH, KW, SH, SW, PH, PW, mode, count_include_pad);

  int errors = 0;
  for (size_t i = 0; i < out_n; ++i) {
    float a = h_y[i];
    float b = h_ref[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}