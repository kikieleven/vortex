// lu (PolyBench-ACC linear-algebra/solvers/lu) — standalone self-checking
// OpenCL port for Vortex.
//
//   for k: A[k][j] /= A[k][k] (j>k);  A[i][j] -= A[i][k]*A[k][j] (i,j>k)
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
//
// Data note: PolyBench's own init `(i*(j+1)+1)/n` makes the matrix exactly
// singular after the first two elimination steps (in float as well as double),
// so that data cannot be used for a self-checking test. This port generates a
// random matrix and adds `n` to the diagonal, which makes it strictly
// diagonally dominant: LU without pivoting then has bounded growth and no
// near-zero pivot.
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

#define KERNEL1_NAME "lu_kernel1"
#define KERNEL2_NAME "lu_kernel2"

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

static float generate() {
  return static_cast<float>(rand()) / RAND_MAX;
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

// A is updated in place, exactly like the kernels.
static void lu_cpu(float* A, int n) {
  for (int k = 0; k < n; ++k) {
    for (int j = k + 1; j < n; ++j) {
      A[k * n + j] = A[k * n + j] / A[k * n + k];
    }
    for (int i = k + 1; i < n; ++i) {
      for (int j = k + 1; j < n; ++j) {
        A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
      }
    }
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel1 = NULL;
cl_kernel kernel2 = NULL;
cl_mem a_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
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

  if (size < 2) {
    printf("Error: size must be at least 2\n");
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
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel1 = CL_CHECK2(clCreateKernel(program, KERNEL1_NAME, &_err));
  kernel2 = CL_CHECK2(clCreateKernel(program, KERNEL2_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 2, sizeof(int), (void *)&n));
  CL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 2, sizeof(int), (void *)&n));

  // Host-side buffers
  std::vector<float> h_a(matrix_size);
  std::vector<float> h_o(matrix_size, 0.0f);

  // Generate a strictly diagonally dominant matrix (see the header note).
  for (size_t i = 0; i < matrix_size; ++i) {
    h_a[i] = generate();
  }
  for (int i = 0; i < n; ++i) {
    h_a[(size_t)i * n + i] += (float)n;
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items: the 1D launch
  // uses 16, the 2D launch uses 4x4; global sizes are rounded up.
  size_t global_1d = ((size + (size_t)16 - 1) / (size_t)16) * (size_t)16;
  size_t local_1d = 16;
  size_t global_2d[2];
  size_t local_2d[2] = {4, 4};
  global_2d[0] = ((size + local_2d[0] - 1) / local_2d[0]) * local_2d[0];
  global_2d[1] = ((size + local_2d[1] - 1) / local_2d[1]) * local_2d[1];

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < n; ++k) {
    CL_CHECK(clSetKernelArg(kernel1, 1, sizeof(int), (void *)&k));
    CL_CHECK(clSetKernelArg(kernel2, 1, sizeof(int), (void *)&k));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 1, NULL, &global_1d, &local_1d, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 2, NULL, global_2d, local_2d, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  lu_cpu(h_a.data(), n);
  int errors = 0;
  for (size_t i = 0; i < matrix_size; ++i) {
    float a = h_o[i];
    float b = h_a[i];
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
