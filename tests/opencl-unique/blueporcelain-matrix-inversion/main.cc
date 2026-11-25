// Blue Porcelain basic/MATRIX_INVERSION — standalone self-checking OpenCL port
// for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/MATRIX_INVERSION/
// matrix_inversion.cu
//
// Upstream has NO verification: it reads a file, writes inv.txt / c.txt and
// always returns 0. This port makes it self-checking: it generates a strictly
// diagonally dominant matrix (so Gauss-Jordan needs no pivoting), computes the
// inverse with the same four update steps on the CPU in the same order, and
// compares every element of the device inverse.
//
// Adaptive steps: 4x4 work-groups instead of 8x8, default n is 32 (upstream
// hard-codes 100), and -n overrides it.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <math.h>

#define TILE 4
#define ABS_TOL 1e-3f
#define REL_TOL 1e-3f

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

  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

// Local deterministic PRNG; see the note in blueporcelain-vectoradd/main.cc.
static uint32_t rng_state = 1;
static uint32_t next_rand() {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
}
static float next_float() {
  return (next_rand() >> 8) * (1.0f / 16777216.0f);
}

// CPU Gauss-Jordan, mirroring the four device kernels step by step.
static void gauss_jordan_cpu(std::vector<float>& A,
                             std::vector<float>& I,
                             int n) {
  for (int i = 0; i < n; ++i) {
    float p = A[i * n + i];
    // nodiag_normalize: row i, y != i
    for (int y = 0; y < n; ++y) {
      if (y != i) {
        I[i * n + y] /= p;
        A[i * n + y] /= p;
      }
    }
    // diag_normalize
    I[i * n + i] /= p;
    A[i * n + i] /= p;
    // gaussjordan: x != i
    for (int x = 0; x < n; ++x) {
      if (x == i) continue;
      float f = A[x * n + i];
      for (int y = 0; y < n; ++y) {
        I[x * n + y] -= I[i * n + y] * f;
        if (y != i)
          A[x * n + y] -= A[i * n + y] * f;
      }
    }
    // set_zero
    for (int x = 0; x < n; ++x)
      if (x != i)
        A[x * n + i] = 0.0f;
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel nodiag_kernel = NULL;
cl_kernel diag_kernel = NULL;
cl_kernel gauss_kernel = NULL;
cl_kernel zero_kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem i_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (nodiag_kernel) clReleaseKernel(nodiag_kernel);
  if (diag_kernel) clReleaseKernel(diag_kernel);
  if (gauss_kernel) clReleaseKernel(gauss_kernel);
  if (zero_kernel) clReleaseKernel(zero_kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (i_memobj) clReleaseMemObject(i_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

int size = 32;

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
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  printf("Matrix size=%d\n", size);

  int n = size;
  size_t nbytes = (size_t)n * n * sizeof(float);

  cl_platform_id platform_id;
  size_t kernel_size;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  i_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  nodiag_kernel = CL_CHECK2(clCreateKernel(program, "nodiag_normalize", &_err));
  diag_kernel = CL_CHECK2(clCreateKernel(program, "diag_normalize", &_err));
  gauss_kernel = CL_CHECK2(clCreateKernel(program, "gaussjordan", &_err));
  zero_kernel = CL_CHECK2(clCreateKernel(program, "set_zero", &_err));

  size_t local_size[2] = {TILE, TILE};
  size_t global_size[2] = {
    ((size_t)(n + TILE - 1) / TILE) * TILE,
    ((size_t)(n + TILE - 1) / TILE) * TILE
  };

  std::vector<float> h_a((size_t)n * n);
  std::vector<float> h_i((size_t)n * n, 0.0f);

  // Strictly diagonally dominant so no pivoting is needed.
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      h_a[r * n + c] = (r == c) ? (float)(n + 1) : (next_float() * 2.0f - 1.0f);
    }
    h_i[r * n + r] = 1.0f;
  }

  std::vector<float> ref_a(h_a);
  std::vector<float> ref_i(h_i);
  gauss_jordan_cpu(ref_a, ref_i, n);

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, i_memobj, CL_TRUE, 0, nbytes, h_i.data(), 0, NULL, NULL));

  printf("Execute the kernels\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i) {
    CL_CHECK(clSetKernelArg(nodiag_kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
    CL_CHECK(clSetKernelArg(nodiag_kernel, 1, sizeof(cl_mem), (void *)&i_memobj));
    CL_CHECK(clSetKernelArg(nodiag_kernel, 2, sizeof(int), &n));
    CL_CHECK(clSetKernelArg(nodiag_kernel, 3, sizeof(int), &i));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, nodiag_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL));

    CL_CHECK(clSetKernelArg(diag_kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
    CL_CHECK(clSetKernelArg(diag_kernel, 1, sizeof(cl_mem), (void *)&i_memobj));
    CL_CHECK(clSetKernelArg(diag_kernel, 2, sizeof(int), &n));
    CL_CHECK(clSetKernelArg(diag_kernel, 3, sizeof(int), &i));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, diag_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL));

    CL_CHECK(clSetKernelArg(gauss_kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
    CL_CHECK(clSetKernelArg(gauss_kernel, 1, sizeof(cl_mem), (void *)&i_memobj));
    CL_CHECK(clSetKernelArg(gauss_kernel, 2, sizeof(int), &n));
    CL_CHECK(clSetKernelArg(gauss_kernel, 3, sizeof(int), &i));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, gauss_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL));

    CL_CHECK(clSetKernelArg(zero_kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
    CL_CHECK(clSetKernelArg(zero_kernel, 1, sizeof(cl_mem), (void *)&i_memobj));
    CL_CHECK(clSetKernelArg(zero_kernel, 2, sizeof(int), &n));
    CL_CHECK(clSetKernelArg(zero_kernel, 3, sizeof(int), &i));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, zero_kernel, 2, NULL, global_size, local_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download inverse buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, i_memobj, CL_TRUE, 0, nbytes, h_i.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      float expected = ref_i[r * n + c];
      float actual = h_i[r * n + c];
      float tol = ABS_TOL + REL_TOL * fabsf(expected);
      if (fabsf(actual - expected) > tol) {
        if (errors < 100)
          printf("*** error: I[%d][%d] expected=%f, actual=%f\n", r, c, expected, actual);
        ++errors;
      }
    }
  }

  if (errors != 0)
    printf("FAILED! - %d errors\n", errors);
  else
    printf("PASSED!\n");

  cleanup();

  return errors;
}