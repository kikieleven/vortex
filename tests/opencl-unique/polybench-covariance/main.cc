// covariance (PolyBench-ACC datamining/covariance) — standalone self-checking
// OpenCL port for Vortex.
//
//   mean -> centre `data` -> m x m covariance matrix
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// m == n (see kernel.cl). `data` is n samples x m variables, row-major.
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the same
// mixed tolerance form used by tests/opencl/{hotspot3D,cfd,lud}. Centred
// products can be negative or near zero.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define MEAN_KERNEL_NAME   "mean_kernel"
#define REDUCE_KERNEL_NAME "reduce_kernel"
#define COVAR_KERNEL_NAME  "covar_kernel"

// Mixed absolute/relative float tolerance (see tests/opencl/hotspot3D/main.cc).
#define REL_TOL 1e-3f
#define ABS_TOL 1e-3f

// PolyBench init_array() value.
#define COVAR_FLOAT_N 1.2f

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

static void covariance_cpu(float* symmat, float* mean, float* data,
                           int m, int n, float float_n) {
  for (int j = 0; j < m; ++j) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      acc += data[i * m + j];
    }
    mean[j] = acc / float_n;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      data[i * m + j] -= mean[j];
    }
  }

  for (int j1 = 0; j1 < m; ++j1) {
    for (int j2 = j1; j2 < m; ++j2) {
      float acc = 0.0f;
      for (int i = 0; i < n; ++i) {
        acc += data[i * m + j1] * data[i * m + j2];
      }
      symmat[j1 * m + j2] = acc;
      symmat[j2 * m + j1] = acc;
    }
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel mean_kernel = NULL;
cl_kernel reduce_kernel = NULL;
cl_kernel covar_kernel = NULL;
cl_mem data_memobj = NULL;
cl_mem mean_memobj = NULL;
cl_mem symmat_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (mean_kernel) clReleaseKernel(mean_kernel);
  if (reduce_kernel) clReleaseKernel(reduce_kernel);
  if (covar_kernel) clReleaseKernel(covar_kernel);
  if (program) clReleaseProgram(program);
  if (data_memobj) clReleaseMemObject(data_memobj);
  if (mean_memobj) clReleaseMemObject(mean_memobj);
  if (symmat_memobj) clReleaseMemObject(symmat_memobj);
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

  const int m = size;
  const int n = size;
  const size_t data_size = (size_t)n * m;
  const size_t vector_size = (size_t)m;
  const size_t symmat_size = (size_t)m * m;
  const size_t data_nbytes = data_size * sizeof(float);
  const size_t vector_nbytes = vector_size * sizeof(float);
  const size_t symmat_nbytes = symmat_size * sizeof(float);

  const float float_n = COVAR_FLOAT_N;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  data_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, data_nbytes, NULL, &_err));
  mean_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, vector_nbytes, NULL, &_err));
  symmat_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, symmat_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  mean_kernel = CL_CHECK2(clCreateKernel(program, MEAN_KERNEL_NAME, &_err));
  reduce_kernel = CL_CHECK2(clCreateKernel(program, REDUCE_KERNEL_NAME, &_err));
  covar_kernel = CL_CHECK2(clCreateKernel(program, COVAR_KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(mean_kernel, 0, sizeof(cl_mem), (void *)&mean_memobj));
  CL_CHECK(clSetKernelArg(mean_kernel, 1, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(mean_kernel, 2, sizeof(float), (void *)&float_n));
  CL_CHECK(clSetKernelArg(mean_kernel, 3, sizeof(int), (void *)&m));
  CL_CHECK(clSetKernelArg(mean_kernel, 4, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), (void *)&mean_memobj));
  CL_CHECK(clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(reduce_kernel, 2, sizeof(int), (void *)&m));
  CL_CHECK(clSetKernelArg(reduce_kernel, 3, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(covar_kernel, 0, sizeof(cl_mem), (void *)&symmat_memobj));
  CL_CHECK(clSetKernelArg(covar_kernel, 1, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(covar_kernel, 2, sizeof(int), (void *)&m));
  CL_CHECK(clSetKernelArg(covar_kernel, 3, sizeof(int), (void *)&n));

  // Host-side input/output buffers
  std::vector<float> h_data(data_size);
  std::vector<float> h_mean(vector_size, 0.0f);
  std::vector<float> h_symmat(symmat_size, 0.0f);
  std::vector<float> h_o(symmat_size, 0.0f);

  // Generate input values
  for (size_t i = 0; i < data_size; ++i) {
    h_data[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, data_memobj, CL_TRUE, 0, data_nbytes, h_data.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items, so the 2D launch
  // uses a 4x4 work-group and the 1D launches use a 16-wide one; global sizes
  // are rounded up to a multiple of the local size.
  const size_t local_1d = 16;
  const size_t local_2d[2] = {4, 4};
  size_t global_1d_m = ((vector_size + local_1d - 1) / local_1d) * local_1d;
  size_t global_2d[2];
  global_2d[0] = ((vector_size + local_2d[0] - 1) / local_2d[0]) * local_2d[0];
  global_2d[1] = ((size_t)n + local_2d[1] - 1) / local_2d[1] * local_2d[1];

  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, mean_kernel, 1, NULL, &global_1d_m, &local_1d, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, reduce_kernel, 2, NULL, global_2d, local_2d, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, covar_kernel, 1, NULL, &global_1d_m, &local_1d, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, symmat_memobj, CL_TRUE, 0, symmat_nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  covariance_cpu(h_symmat.data(), h_mean.data(), h_data.data(), m, n, float_n);
  int errors = 0;
  for (size_t i = 0; i < symmat_size; ++i) {
    float a = h_o[i];
    float b = h_symmat[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: symmat[%d] expected=%f, actual=%f (diff=%g)\n",
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
