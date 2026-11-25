// adi (PolyBench-ACC stencils/adi) — standalone self-checking OpenCL port for
// Vortex.
//
// Per time step: row elimination -> row scale -> row back-substitution ->
// column elimination -> column scale -> column back-substitution.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// X, A and B are n x n.
//
// Data: PolyBench's own init is used (X/A/B are positive and the `B`
// denominators stay away from zero at the default sizes); a random matrix would
// make the repeated `... / B` recurrences ill-conditioned.
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the same
// mixed tolerance form used by tests/opencl/{hotspot3D,cfd,lud}.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL1_NAME "adi_kernel1"
#define KERNEL2_NAME "adi_kernel2"
#define KERNEL3_NAME "adi_kernel3"
#define KERNEL4_NAME "adi_kernel4"
#define KERNEL5_NAME "adi_kernel5"
#define KERNEL6_NAME "adi_kernel6"

// Mixed absolute/relative float tolerance (see tests/opencl/hotspot3D/main.cc).
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

static void adi_cpu(float* X, float* A, float* B, int n, int tsteps) {
  for (int t = 0; t < tsteps; ++t) {
    for (int i1 = 0; i1 < n; ++i1) {
      for (int i2 = 1; i2 < n; ++i2) {
        X[i1 * n + i2] = X[i1 * n + i2]
                       - X[i1 * n + (i2 - 1)] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
        B[i1 * n + i2] = B[i1 * n + i2]
                       - A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
      }
    }

    for (int i1 = 0; i1 < n; ++i1) {
      X[i1 * n + (n - 1)] = X[i1 * n + (n - 1)] / B[i1 * n + (n - 1)];
    }

    for (int i1 = 0; i1 < n; ++i1) {
      for (int i2 = 0; i2 < n - 2; ++i2) {
        X[i1 * n + (n - i2 - 2)] =
            (X[i1 * n + (n - 2 - i2)] - X[i1 * n + (n - 2 - i2 - 1)] * A[i1 * n + (n - i2 - 3)])
            / B[i1 * n + (n - 3 - i2)];
      }
    }

    for (int i1 = 1; i1 < n; ++i1) {
      for (int i2 = 0; i2 < n; ++i2) {
        X[i1 * n + i2] = X[i1 * n + i2]
                       - X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
        B[i1 * n + i2] = B[i1 * n + i2]
                       - A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
      }
    }

    for (int i2 = 0; i2 < n; ++i2) {
      X[(n - 1) * n + i2] = X[(n - 1) * n + i2] / B[(n - 1) * n + i2];
    }

    for (int i1 = 0; i1 < n - 2; ++i1) {
      for (int i2 = 0; i2 < n; ++i2) {
        X[(n - 2 - i1) * n + i2] =
            (X[(n - 2 - i1) * n + i2] - X[(n - i1 - 3) * n + i2] * A[(n - 3 - i1) * n + i2])
            / B[(n - 2 - i1) * n + i2];
      }
    }
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernels[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem x_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  for (int i = 0; i < 6; ++i) {
    if (kernels[i]) clReleaseKernel(kernels[i]);
  }
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (x_memobj) clReleaseMemObject(x_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 16;
int tsteps = 2;

static void show_usage() {
  printf("Usage: [-n size] [-t tsteps] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 't':
      tsteps = atoi(optarg);
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

  printf("float workload size=%d tsteps=%d\n", size, tsteps);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 3 || tsteps < 1) {
    printf("Error: size must be at least 3 and tsteps at least 1\n");
    return -1;
  }

  const int n = size;
  const size_t matrix_size = (size_t)n * n;
  const size_t matrix_nbytes = matrix_size * sizeof(float);

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));
  x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernels[0] = CL_CHECK2(clCreateKernel(program, KERNEL1_NAME, &_err));
  kernels[1] = CL_CHECK2(clCreateKernel(program, KERNEL2_NAME, &_err));
  kernels[2] = CL_CHECK2(clCreateKernel(program, KERNEL3_NAME, &_err));
  kernels[3] = CL_CHECK2(clCreateKernel(program, KERNEL4_NAME, &_err));
  kernels[4] = CL_CHECK2(clCreateKernel(program, KERNEL5_NAME, &_err));
  kernels[5] = CL_CHECK2(clCreateKernel(program, KERNEL6_NAME, &_err));

  // kernels 1,2,3,5: (A, B, X, n);  kernels 4,6: (A, B, X, i1, n)
  for (int i = 0; i < 3; ++i) {
    CL_CHECK(clSetKernelArg(kernels[i], 0, sizeof(cl_mem), (void *)&a_memobj));
    CL_CHECK(clSetKernelArg(kernels[i], 1, sizeof(cl_mem), (void *)&b_memobj));
    CL_CHECK(clSetKernelArg(kernels[i], 2, sizeof(cl_mem), (void *)&x_memobj));
    CL_CHECK(clSetKernelArg(kernels[i], 3, sizeof(int), (void *)&n));
  }
  CL_CHECK(clSetKernelArg(kernels[2], 1, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernels[4], 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernels[4], 1, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernels[4], 2, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernels[4], 3, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(kernels[3], 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernels[3], 1, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernels[3], 2, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernels[3], 4, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(kernels[5], 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernels[5], 1, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernels[5], 2, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernels[5], 4, sizeof(int), (void *)&n));

  // Host-side buffers, initialised with PolyBench's init_array formulas.
  std::vector<float> h_x(matrix_size);
  std::vector<float> h_a(matrix_size);
  std::vector<float> h_b(matrix_size);
  std::vector<float> h_o(matrix_size, 0.0f);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      h_x[i * n + j] = ((float)(i * (j + 1) + 1)) / n;
      h_a[i * n + j] = ((float)(i * (j + 2) + 2)) / n;
      h_b[i * n + j] = ((float)(i * (j + 3) + 3)) / n;
    }
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, matrix_nbytes, h_b.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, matrix_nbytes, h_x.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items; every stage is
  // 1D over n with a 16-wide work-group and an upward-rounded global size.
  size_t local_work_size = 16;
  size_t global_work_size = ((size + local_work_size - 1) / local_work_size) * local_work_size;

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < tsteps; ++t) {
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernels[0], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernels[1], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernels[2], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));

    for (int i1 = 1; i1 < n; ++i1) {
      CL_CHECK(clSetKernelArg(kernels[3], 3, sizeof(int), (void *)&i1));
      CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernels[3], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    }

    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernels[4], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));

    for (int i1 = 0; i1 < n - 2; ++i1) {
      CL_CHECK(clSetKernelArg(kernels[5], 3, sizeof(int), (void *)&i1));
      CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernels[5], 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    }
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, x_memobj, CL_TRUE, 0, matrix_nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  adi_cpu(h_x.data(), h_a.data(), h_b.data(), n, tsteps);
  int errors = 0;
  for (size_t i = 0; i < matrix_size; ++i) {
    float a = h_o[i];
    float b = h_x[i];
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
