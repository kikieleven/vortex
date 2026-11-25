// conv2d_relu (Ventus-OpenCL-Testcase, _get_case/AIops/conv_fused) —
// standalone self-checking OpenCL port for Vortex.
//
//   out[oc,oh,ow] = max(0, sum_{ic,kh,kw} in[ic,ih,iw] * w[oc,ic,kh,kw])
//   ih = oh*SH - PH + kh,  iw = ow*SW - PW + kw          (no bias term)
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/conv_fused/
// — the directory has only conv_fused.cl, so the host and CPU reference below
// are new. The convolution body and the fused ReLU are the upstream kernel.
//
// Notes:
//   * the launch keeps the upstream 3D NDRange {OC, OH, OW}; local {1,4,4},
//     global rounded up, kernel guard drops the extra work-items.
//   * inputs come from a private deterministic LCG in [-1,1) (porting guide
//     §13.2). Signed inputs make the ReLU epilogue actually fire; with
//     non-negative inputs every output would pass through unchanged and the
//     fused path would not be exercised.
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL: the
//     ReLU makes half the outputs zero and the rest signed, so ULP is unusable
//     (porting guide §4.7).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "conv2d_relu"

#define REL_TOL 1e-3f
#define ABS_TOL 1e-3f

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

// Same loop nesting, same accumulation order, same boundary test as kernel.cl.
static void conv2d_relu_cpu(float* out, const float* in, const float* w,
                            int IC, int IH, int IW,
                            int OC, int KH, int KW,
                            int OH, int OW) {
  const int SH = 1, SW = 1, PH = 1, PW = 1;
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        float value = 0.0f;
        for (int ic = 0; ic < IC; ++ic) {
          for (int kh = 0; kh < KH; ++kh) {
            const int ih = oh * SH - PH + kh;
            if (ih < 0 || ih >= IH) continue;
            for (int kw = 0; kw < KW; ++kw) {
              const int iw = ow * SW - PW + kw;
              if (iw < 0 || iw >= IW) continue;
              const int in_idx = (ic * IH + ih) * IW + iw;
              const int w_idx = (oc * IC + ic) * KH * KW + kh * KW + kw;
              value += in[in_idx] * w[w_idx];
            }
          }
        }
        out[(oc * OH + oh) * OW + ow] = fmaxf(0.0f, value);
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
cl_mem in_memobj = NULL;
cl_mem w_memobj = NULL;
cl_mem o_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (in_memobj) clReleaseMemObject(in_memobj);
  if (w_memobj) clReleaseMemObject(w_memobj);
  if (o_memobj) clReleaseMemObject(o_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 16;

static void show_usage() {
  printf("Usage: [-n size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
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

  printf("float workload size=%d\n", size);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 1) {
    printf("Error: size must be at least 1\n");
    return -1;
  }

  const int IC = 3;
  const int IH = size;
  const int IW = size;
  const int OC = 8;
  const int KH = 3;
  const int KW = 3;
  const int SH = 1;
  const int SW = 1;
  const int PH = 1;
  const int PW = 1;
  const int OH = (IH + 2 * PH - KH) / SH + 1;
  const int OW = (IW + 2 * PW - KW) / SW + 1;
  if (OH <= 0 || OW <= 0) {
    printf("Error: invalid output size OH=%d OW=%d\n", OH, OW);
    return -1;
  }

  const size_t in_n = (size_t)IC * IH * IW;
  const size_t w_n = (size_t)OC * IC * KH * KW;
  const size_t out_n = (size_t)OC * OH * OW;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, in_n * sizeof(float), NULL, &_err));
  w_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w_n * sizeof(float), NULL, &_err));
  o_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_n * sizeof(float), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int arg = 0;
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&in_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&w_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&o_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&IC));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&IH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&IW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&OC));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&KH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&KW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&OH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&OW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&SH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&SW));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&PH));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&PW));

  std::vector<float> h_in(in_n), h_w(w_n);
  std::vector<float> h_o(out_n, 0.0f), h_ref(out_n);
  for (size_t i = 0; i < in_n; ++i)
    h_in[i] = generate_pm1();
  for (size_t i = 0; i < w_n; ++i)
    h_w[i] = generate_pm1();

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, in_n * sizeof(float), h_in.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_memobj, CL_TRUE, 0, w_n * sizeof(float), h_w.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t local_work_size[3] = {1, 4, 4};
  size_t global_work_size[3];
  global_work_size[0] = round_up((size_t)OC, local_work_size[0]);
  global_work_size[1] = round_up((size_t)OH, local_work_size[1]);
  global_work_size[2] = round_up((size_t)OW, local_work_size[2]);
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, o_memobj, CL_TRUE, 0, out_n * sizeof(float), h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  conv2d_relu_cpu(h_ref.data(), h_in.data(), h_w.data(),
                  IC, IH, IW, OC, KH, KW, OH, OW);

  int errors = 0;
  for (size_t i = 0; i < out_n; ++i) {
    float a = h_o[i];
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