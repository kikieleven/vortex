// SHOC GEMM (level1/gemm) — standalone self-checking OpenCL port for Vortex,
// single precision only.
//
// Source:
//   third_benchmark/shoc/src/opencl/level1/gemm/gemmN.cl
//   third_benchmark/shoc/src/opencl/level1/gemm/GEMM.cpp:143-160, 249-260
//     (problem size, alpha/beta, random data)
//
// SHOC has no CPU reference and does not verify GEMM output; this port adds a
// host reference and a mixed abs/rel comparison.
//
// Kernel: SHOC's sgemmNN is a Volkov register-tiled kernel built for a
// 64-work-item group (16x4 local ids, 16 accumulators per thread). Vortex caps
// a work-group at 16, so kernel.cl re-blocks the same blocked-GEMM algorithm
// for 4x4 work-items / 16x16 output tiles. See kernel.cl for the mapping.
//
// Data: srand48(13579862), A and B entries in [0.5, 2.0), C = 0,
// alpha = 1, beta = -1 (GEMM.cpp:153-154, 249-260). Both sides accumulate over
// k in ascending order, so only FMA contraction can perturb the result; the
// tolerance is 1e-3 relative + 1e-3 absolute.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "sgemmNN"

#define TILE 16
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

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
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

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem c_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (c_memobj) clReleaseMemObject(c_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 32;

static void show_usage() {
  printf("Usage: [-n matrix size] [-h: help]\n");
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

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < TILE || (size % TILE) != 0) {
    printf("Error: size must be a positive multiple of %d\n", TILE);
    return -1;
  }

  const int n = size;
  const size_t matrix_size = (size_t)n * n;
  const size_t matrix_nbytes = matrix_size * sizeof(float);

  const float alpha = 1.0f;   // GEMM.cpp:153
  const float beta = -1.0f;   // GEMM.cpp:154

  // GEMM.cpp:249-260: fixed seed, A/B in [0.5, 2.0), C = 0.
  std::vector<float> h_a(matrix_size), h_b(matrix_size), h_c(matrix_size, 0.0f), h_ref(matrix_size);
  srand48(13579862);
  for (size_t i = 0; i < matrix_size; ++i)
    h_a[i] = (float)(0.5 + drand48() * 1.5);
  for (size_t i = 0; i < matrix_size; ++i)
    h_b[i] = (float)(0.5 + drand48() * 1.5);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  printf("Create program from kernel source\n");
  size_t kernel_size = 0;
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_nbytes, NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), (void *)&n));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void *)&n));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&c_memobj));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void *)&n));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), (void *)&n));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(float), (void *)&alpha));
  CL_CHECK(clSetKernelArg(kernel, 8, sizeof(float), (void *)&beta));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, matrix_nbytes, h_b.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, c_memobj, CL_TRUE, 0, matrix_nbytes, h_c.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t global_work_size[2] = {(size_t)n, (size_t)n};
  const size_t local_work_size[2] = {4, 4};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                  global_work_size, local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  std::vector<float> h_out(matrix_size);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, c_memobj, CL_TRUE, 0, matrix_nbytes,
                               h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Host reference: same NN product, accumulation over k ascending, then the
  // alpha/beta epilogue with the initial C (zero).
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int kk = 0; kk < n; ++kk)
        acc += h_a[(size_t)i * n + kk] * h_b[(size_t)kk * n + j];
      h_ref[(size_t)i * n + j] = alpha * acc + beta * h_c[(size_t)i * n + j];
    }
  }

  int errors = 0;
  for (size_t i = 0; i < matrix_size; ++i) {
    float expected = h_ref[i];
    float actual = h_out[i];
    float tol = REL_TOL * fabsf(expected) + ABS_TOL;
    if (fabsf(actual - expected) > tol) {
      if (errors < 100) {
        printf("*** error: [%d,%d] expected=%f, actual=%f (diff=%g)\n",
               (int)(i / n), (int)(i % n), expected, actual,
               fabsf(actual - expected));
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
