// BiCG (PolyBench-ACC linear-algebra/kernels/bicg) — standalone self-checking
// OpenCL port for Vortex.
//
//   q = A   * p
//   s = A^T * r
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// A is n x n (see kernel.cl). The host reference evaluates the same two sums in
// the same order as PolyBench's CPU version.
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the same
// mixed tolerance form used by tests/opencl/{hotspot3D,cfd,lud}. The pure-ULP
// comparator needs a small non-negative result; here each entry is an n-term
// accumulation, so the rounding differences are relative.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL1_NAME "bicg_kernel1"
#define KERNEL2_NAME "bicg_kernel2"

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

static void bicg_cpu(float* q, float* s, const float* A,
                     const float* p, const float* r, int n) {
  for (int i = 0; i < n; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      acc += A[i * n + j] * p[j];
    }
    q[i] = acc;
  }

  for (int j = 0; j < n; ++j) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      acc += A[i * n + j] * r[i];
    }
    s[j] = acc;
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel1 = NULL;
cl_kernel kernel2 = NULL;
cl_mem a_memobj = NULL;
cl_mem p_memobj = NULL;
cl_mem r_memobj = NULL;
cl_mem q_memobj = NULL;
cl_mem s_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (p_memobj) clReleaseMemObject(p_memobj);
  if (r_memobj) clReleaseMemObject(r_memobj);
  if (q_memobj) clReleaseMemObject(q_memobj);
  if (s_memobj) clReleaseMemObject(s_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 128;

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

  const int n = size;
  const size_t matrix_size = (size_t)n * n;
  const size_t vector_size = (size_t)n;
  const size_t matrix_nbytes = matrix_size * sizeof(float);
  const size_t vector_nbytes = vector_size * sizeof(float);

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_nbytes, NULL, &_err));
  p_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));
  r_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));
  q_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, vector_nbytes, NULL, &_err));
  s_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, vector_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel1 = CL_CHECK2(clCreateKernel(program, KERNEL1_NAME, &_err));
  kernel2 = CL_CHECK2(clCreateKernel(program, KERNEL2_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&p_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&q_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 3, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&r_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&s_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 3, sizeof(int), (void *)&n));

  // Host-side input/output buffers
  std::vector<float> h_a(matrix_size);
  std::vector<float> h_p(vector_size);
  std::vector<float> h_r(vector_size);
  std::vector<float> h_q(vector_size, 0.0f);
  std::vector<float> h_s(vector_size, 0.0f);
  std::vector<float> h_q_o(vector_size, 0.0f);
  std::vector<float> h_s_o(vector_size, 0.0f);

  // Generate input values
  for (size_t i = 0; i < matrix_size; ++i) {
    h_a[i] = generate();
  }
  for (size_t i = 0; i < vector_size; ++i) {
    h_p[i] = generate();
    h_r[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, p_memobj, CL_TRUE, 0, vector_nbytes, h_p.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, r_memobj, CL_TRUE, 0, vector_nbytes, h_r.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items, so use a 16-wide
  // 1D work-group (PolyBench's original 32x8 is not portable here) and round
  // the global size up to a multiple of the local size.
  size_t local_work_size = 16;
  size_t global_work_size = ((size + local_work_size - 1) / local_work_size) * local_work_size;
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 1, NULL, &global_work_size, &local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, q_memobj, CL_TRUE, 0, vector_nbytes, h_q_o.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, s_memobj, CL_TRUE, 0, vector_nbytes, h_s_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  bicg_cpu(h_q.data(), h_s.data(), h_a.data(), h_p.data(), h_r.data(), n);
  int errors = 0;
  for (size_t i = 0; i < vector_size; ++i) {
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
  for (size_t i = 0; i < vector_size; ++i) {
    float a = h_s_o[i];
    float b = h_s[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: s[%d] expected=%f, actual=%f (diff=%g)\n",
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
