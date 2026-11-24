// gemver (PolyBench-ACC linear-algebra/kernels/gemver) — standalone
// self-checking OpenCL port for Vortex.
//
//   A += u1*v1^T + u2*v2^T
//   x += beta * A^T * y + z
//   w += alpha * A * x
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// x and w are input/output: their initial values are uploaded and the CPU
// reference accumulates on top.
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

#define KERNEL1_NAME "gemver_kernel1"
#define KERNEL2_NAME "gemver_kernel2"
#define KERNEL3_NAME "gemver_kernel3"

// Mixed absolute/relative float tolerance (see tests/opencl/hotspot3D/main.cc).
#define REL_TOL 1e-3f
#define ABS_TOL 1e-3f

// PolyBench init_array() values.
#define GEMVER_ALPHA 43532.0f
#define GEMVER_BETA  12313.0f

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

// A, x and w are updated in place, exactly like the kernels.
static void gemver_cpu(float* A, const float* u1, const float* v1,
                       const float* u2, const float* v2,
                       float* w, float* x, const float* y, const float* z,
                       int n, float alpha, float beta) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];
    }
  }

  for (int i = 0; i < n; ++i) {
    float acc = x[i];
    for (int j = 0; j < n; ++j) {
      acc += beta * A[j * n + i] * y[j];
    }
    acc += z[i];
    x[i] = acc;
  }

  for (int i = 0; i < n; ++i) {
    float acc = w[i];
    for (int j = 0; j < n; ++j) {
      acc += alpha * A[i * n + j] * x[j];
    }
    w[i] = acc;
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
cl_mem u1_memobj = NULL;
cl_mem v1_memobj = NULL;
cl_mem u2_memobj = NULL;
cl_mem v2_memobj = NULL;
cl_mem w_memobj = NULL;
cl_mem x_memobj = NULL;
cl_mem y_memobj = NULL;
cl_mem z_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (kernel3) clReleaseKernel(kernel3);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (u1_memobj) clReleaseMemObject(u1_memobj);
  if (v1_memobj) clReleaseMemObject(v1_memobj);
  if (u2_memobj) clReleaseMemObject(u2_memobj);
  if (v2_memobj) clReleaseMemObject(v2_memobj);
  if (w_memobj) clReleaseMemObject(w_memobj);
  if (x_memobj) clReleaseMemObject(x_memobj);
  if (y_memobj) clReleaseMemObject(y_memobj);
  if (z_memobj) clReleaseMemObject(z_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 64;

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

  const float alpha = GEMVER_ALPHA;
  const float beta = GEMVER_BETA;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));
  u1_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));
  v1_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));
  u2_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));
  v2_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));
  w_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, vector_nbytes, NULL, &_err));
  x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, vector_nbytes, NULL, &_err));
  y_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));
  z_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vector_nbytes, NULL, &_err));

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
  CL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&v1_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&v2_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&u1_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void *)&u2_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 5, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&y_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void *)&z_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 4, sizeof(float), (void *)&beta));
  CL_CHECK(clSetKernelArg(kernel2, 5, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 1, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 2, sizeof(cl_mem), (void *)&w_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 3, sizeof(float), (void *)&alpha));
  CL_CHECK(clSetKernelArg(kernel3, 4, sizeof(int), (void *)&n));

  // Host-side buffers
  std::vector<float> h_a(matrix_size);
  std::vector<float> h_u1(vector_size), h_v1(vector_size);
  std::vector<float> h_u2(vector_size), h_v2(vector_size);
  std::vector<float> h_w(vector_size), h_x(vector_size);
  std::vector<float> h_y(vector_size), h_z(vector_size);
  std::vector<float> h_w_o(vector_size, 0.0f), h_x_o(vector_size, 0.0f);

  // Generate input values
  for (size_t i = 0; i < matrix_size; ++i) {
    h_a[i] = generate();
  }
  for (size_t i = 0; i < vector_size; ++i) {
    h_u1[i] = generate();
    h_v1[i] = generate();
    h_u2[i] = generate();
    h_v2[i] = generate();
    h_w[i] = generate();
    h_x[i] = generate();
    h_y[i] = generate();
    h_z[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, u1_memobj, CL_TRUE, 0, vector_nbytes, h_u1.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, v1_memobj, CL_TRUE, 0, vector_nbytes, h_v1.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, u2_memobj, CL_TRUE, 0, vector_nbytes, h_u2.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, v2_memobj, CL_TRUE, 0, vector_nbytes, h_v2.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_memobj, CL_TRUE, 0, vector_nbytes, h_w.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, vector_nbytes, h_x.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, y_memobj, CL_TRUE, 0, vector_nbytes, h_y.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, z_memobj, CL_TRUE, 0, vector_nbytes, h_z.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items: the 2D launch
  // uses 4x4, the 1D launches use 16; global sizes are rounded up.
  size_t global_2d[2];
  size_t local_2d[2] = {4, 4};
  global_2d[0] = ((size + local_2d[0] - 1) / local_2d[0]) * local_2d[0];
  global_2d[1] = ((size + local_2d[1] - 1) / local_2d[1]) * local_2d[1];
  size_t local_1d = 16;
  size_t global_1d = ((size + local_1d - 1) / local_1d) * local_1d;

  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 2, NULL, global_2d, local_2d, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 1, NULL, &global_1d, &local_1d, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel3, 1, NULL, &global_1d, &local_1d, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, x_memobj, CL_TRUE, 0, vector_nbytes, h_x_o.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, w_memobj, CL_TRUE, 0, vector_nbytes, h_w_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  gemver_cpu(h_a.data(), h_u1.data(), h_v1.data(), h_u2.data(), h_v2.data(),
             h_w.data(), h_x.data(), h_y.data(), h_z.data(), n, alpha, beta);
  int errors = 0;
  for (size_t i = 0; i < vector_size; ++i) {
    float a = h_x_o[i];
    float b = h_x[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: x[%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  for (size_t i = 0; i < vector_size; ++i) {
    float a = h_w_o[i];
    float b = h_w[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: w[%d] expected=%f, actual=%f (diff=%g)\n",
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
