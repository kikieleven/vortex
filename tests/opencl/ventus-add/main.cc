// add (Ventus-OpenCL-Testcase, _get_case/AIops/add) — standalone
// self-checking OpenCL port for Vortex.
//
//   out[c,y,x] = a[c,y,x] + b[c,y,x]      idx = c*(H*W) + y*W + x
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/add/ — the
// directory has only add.cl (main.py is empty), so the host and CPU reference
// below are new. They follow the kernel exactly: same index expression, same
// single float addition.
//
// Notes:
//   * the launch keeps the upstream 3D NDRange {C, H, W}; the global size is
//     rounded up to the 16-work-item cap with local {1,4,4} and the kernel's
//     guard drops the extra work-items (verified by ventus-conv in batch one).
//   * inputs come from a private deterministic LCG instead of rand(): PoCL
//     shares the process-wide rand() state, which makes the simx/rtlsim
//     retired-instruction counts differ between runs (porting guide §13.2).
//   * comparison is **bit exact**. A single IEEE float addition is
//     deterministic and both sides evaluate the same expression on the same
//     operands, so requiring identical bit patterns (including the sign of
//     zero) is legitimate here and stricter than a tolerance.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "add3d"

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

static bool same_bits(float a, float b) {
  union fi_t { float f; int32_t i; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  return fa.i == fb.i;
}

static size_t round_up(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem o_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (o_memobj) clReleaseMemObject(o_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int dim_c = 4;
int dim_h = 8;
int dim_w = 12;

static void show_usage() {
  printf("Usage: [-c C] [-y H] [-x W] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "c:y:x:h")) != -1) {
    switch (c) {
    case 'c':
      dim_c = atoi(optarg);
      break;
    case 'y':
      dim_h = atoi(optarg);
      break;
    case 'x':
      dim_w = atoi(optarg);
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

  printf("float workload C=%d H=%d W=%d\n", dim_c, dim_h, dim_w);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (dim_c < 1 || dim_h < 1 || dim_w < 1) {
    printf("Error: all dimensions must be at least 1\n");
    return -1;
  }

  const int C = dim_c;
  const int H = dim_h;
  const int W = dim_w;
  const size_t n = (size_t)C * H * W;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &_err));
  o_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int arg = 0;
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&o_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&C));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&H));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&W));

  std::vector<float> h_a(n), h_b(n), h_o(n, 0.0f), h_ref(n);
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = generate_pm1();
    h_b[i] = generate_pm1();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, n * sizeof(float), h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, n * sizeof(float), h_b.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t local_work_size[3] = {1, 4, 4};
  size_t global_work_size[3];
  global_work_size[0] = round_up((size_t)C, local_work_size[0]);
  global_work_size[1] = round_up((size_t)H, local_work_size[1]);
  global_work_size[2] = round_up((size_t)W, local_work_size[2]);
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, o_memobj, CL_TRUE, 0, n * sizeof(float), h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  for (size_t i = 0; i < n; ++i)
    h_ref[i] = h_a[i] + h_b[i];

  int errors = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!same_bits(h_o[i], h_ref[i])) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n",
               (int)i, h_ref[i], h_o[i]);
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