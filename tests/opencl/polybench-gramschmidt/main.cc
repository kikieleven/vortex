// gramschmidt (PolyBench-ACC linear-algebra/solvers/gramschmidt) — standalone
// self-checking OpenCL port for Vortex.
//
//   for k: R[k][k] = |A[:,k]|; Q[:,k] = A[:,k]/R[k][k];
//          for j>k: R[k][j] = Q[:,k].A[:,j]; A[:,j] -= Q[:,k]*R[k][j]
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// A is n x n and is updated in place, so its initial values are uploaded.
//
// R's lower triangle is never written by the algorithm and keeps whatever the
// host supplied; Q is fully overwritten but is also uploaded with known values,
// so a kernel that silently skipped a column would still be caught.
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the same
// mixed tolerance form used by tests/opencl/{hotspot3D,cfd,lud}. A and R can be
// negative and can pass through zero.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL1_NAME "gramschmidt_kernel1"
#define KERNEL2_NAME "gramschmidt_kernel2"
#define KERNEL3_NAME "gramschmidt_kernel3"

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

// A, R and Q are updated in place, exactly like the kernels.
static void gramschmidt_cpu(float* a, float* r, float* q, int n) {
  for (int k = 0; k < n; ++k) {
    float nrm = 0.0f;
    for (int i = 0; i < n; ++i) {
      nrm += a[i * n + k] * a[i * n + k];
    }
    r[k * n + k] = sqrtf(nrm);

    for (int i = 0; i < n; ++i) {
      q[i * n + k] = a[i * n + k] / r[k * n + k];
    }

    for (int j = k + 1; j < n; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < n; ++i) {
        acc += q[i * n + k] * a[i * n + j];
      }
      r[k * n + j] = acc;

      for (int i = 0; i < n; ++i) {
        a[i * n + j] -= q[i * n + k] * r[k * n + j];
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
cl_kernel kernel3 = NULL;
cl_mem a_memobj = NULL;
cl_mem r_memobj = NULL;
cl_mem q_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (kernel3) clReleaseKernel(kernel3);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (r_memobj) clReleaseMemObject(r_memobj);
  if (q_memobj) clReleaseMemObject(q_memobj);
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
  r_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));
  q_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel1 = CL_CHECK2(clCreateKernel(program, KERNEL1_NAME, &_err));
  kernel2 = CL_CHECK2(clCreateKernel(program, KERNEL2_NAME, &_err));
  kernel3 = CL_CHECK2(clCreateKernel(program, KERNEL3_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&r_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&q_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 4, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&r_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&q_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 4, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 1, sizeof(cl_mem), (void *)&r_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 2, sizeof(cl_mem), (void *)&q_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 4, sizeof(int), (void *)&n));

  // Host-side working buffers
  std::vector<float> h_a(matrix_size);
  std::vector<float> h_r(matrix_size);
  std::vector<float> h_q(matrix_size);
  std::vector<float> h_a_o(matrix_size, 0.0f);
  std::vector<float> h_r_o(matrix_size, 0.0f);
  std::vector<float> h_q_o(matrix_size, 0.0f);

  // Generate input values
  for (size_t i = 0; i < matrix_size; ++i) {
    h_a[i] = generate();
    h_r[i] = generate();
    h_q[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, r_memobj, CL_TRUE, 0, matrix_nbytes, h_r.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, q_memobj, CL_TRUE, 0, matrix_nbytes, h_q.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items. kernel1 is a
  // single work-item; kernel2/kernel3 use a 16-wide work-group with the global
  // size rounded up to a multiple of it.
  size_t local_single = 1;
  size_t global_single = 1;
  size_t local_work_size = 16;
  size_t global_work_size = ((size + local_work_size - 1) / local_work_size) * local_work_size;

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int k = 0; k < n; ++k) {
    CL_CHECK(clSetKernelArg(kernel1, 3, sizeof(int), (void *)&k));
    CL_CHECK(clSetKernelArg(kernel2, 3, sizeof(int), (void *)&k));
    CL_CHECK(clSetKernelArg(kernel3, 3, sizeof(int), (void *)&k));

    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 1, NULL, &global_single, &local_single, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel3, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a_o.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, r_memobj, CL_TRUE, 0, matrix_nbytes, h_r_o.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, q_memobj, CL_TRUE, 0, matrix_nbytes, h_q_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  gramschmidt_cpu(h_a.data(), h_r.data(), h_q.data(), n);
  int errors = 0;
  for (size_t i = 0; i < matrix_size; ++i) {
    float a = h_a_o[i];
    float b = h_a[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: a[%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  for (size_t i = 0; i < matrix_size; ++i) {
    float a = h_r_o[i];
    float b = h_r[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: r[%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  for (size_t i = 0; i < matrix_size; ++i) {
    float a = h_q_o[i];
    float b = h_q[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: q[%d] expected=%f, actual=%f (diff=%g)\n",
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
