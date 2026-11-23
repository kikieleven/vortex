// correlation (PolyBench-ACC datamining/correlation) — standalone
// self-checking OpenCL port for Vortex.
//
//   mean -> stddev -> centre+normalise `data` -> m x m correlation matrix
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// m == n (see kernel.cl). `data` is n samples x m variables, row-major.
//
// After the correlation kernel the host writes symmat[m-1][m-1] = 1, exactly as
// PolyBench's OpenCL host does (the kernel only visits j1 < m-1).
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the same
// mixed tolerance form used by tests/opencl/{hotspot3D,cfd,lud}. Correlation
// entries are in [-1, 1] and can be negative.

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
#define STD_KERNEL_NAME    "std_kernel"
#define REDUCE_KERNEL_NAME "reduce_kernel"
#define CORR_KERNEL_NAME   "corr_kernel"

// Mixed absolute/relative float tolerance (see tests/opencl/hotspot3D/main.cc).
#define REL_TOL 1e-3f
#define ABS_TOL 1e-3f

// PolyBench init_array()/kernel values.
#define CORR_FLOAT_N 1.2f
#define CORR_EPS     0.1f

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

static void correlation_cpu(float* symmat, float* mean, float* std,
                            float* data, int m, int n,
                            float float_n, float eps) {
  for (int j = 0; j < m; ++j) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      acc += data[i * m + j];
    }
    mean[j] = acc / float_n;
  }

  for (int j = 0; j < m; ++j) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      float d = data[i * m + j] - mean[j];
      acc += d * d;
    }
    acc /= float_n;
    acc = sqrtf(acc);
    std[j] = (acc <= eps) ? 1.0f : acc;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      float d = data[i * m + j] - mean[j];
      data[i * m + j] = d / (sqrtf(float_n) * std[j]);
    }
  }

  for (int j1 = 0; j1 < m - 1; ++j1) {
    symmat[j1 * m + j1] = 1.0f;
    for (int j2 = j1 + 1; j2 < m; ++j2) {
      float acc = 0.0f;
      for (int i = 0; i < n; ++i) {
        acc += data[i * m + j1] * data[i * m + j2];
      }
      symmat[j1 * m + j2] = acc;
      symmat[j2 * m + j1] = acc;
    }
  }
  symmat[(m - 1) * m + (m - 1)] = 1.0f;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel mean_kernel = NULL;
cl_kernel std_kernel = NULL;
cl_kernel reduce_kernel = NULL;
cl_kernel corr_kernel = NULL;
cl_mem data_memobj = NULL;
cl_mem mean_memobj = NULL;
cl_mem std_memobj = NULL;
cl_mem symmat_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (mean_kernel) clReleaseKernel(mean_kernel);
  if (std_kernel) clReleaseKernel(std_kernel);
  if (reduce_kernel) clReleaseKernel(reduce_kernel);
  if (corr_kernel) clReleaseKernel(corr_kernel);
  if (program) clReleaseProgram(program);
  if (data_memobj) clReleaseMemObject(data_memobj);
  if (mean_memobj) clReleaseMemObject(mean_memobj);
  if (std_memobj) clReleaseMemObject(std_memobj);
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

  if (size < 3) {
    printf("Error: size must be at least 3\n");
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

  const float float_n = CORR_FLOAT_N;
  const float eps = CORR_EPS;

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
  std_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, vector_nbytes, NULL, &_err));
  symmat_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, symmat_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  mean_kernel = CL_CHECK2(clCreateKernel(program, MEAN_KERNEL_NAME, &_err));
  std_kernel = CL_CHECK2(clCreateKernel(program, STD_KERNEL_NAME, &_err));
  reduce_kernel = CL_CHECK2(clCreateKernel(program, REDUCE_KERNEL_NAME, &_err));
  corr_kernel = CL_CHECK2(clCreateKernel(program, CORR_KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(mean_kernel, 0, sizeof(cl_mem), (void *)&mean_memobj));
  CL_CHECK(clSetKernelArg(mean_kernel, 1, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(mean_kernel, 2, sizeof(float), (void *)&float_n));
  CL_CHECK(clSetKernelArg(mean_kernel, 3, sizeof(int), (void *)&m));
  CL_CHECK(clSetKernelArg(mean_kernel, 4, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(std_kernel, 0, sizeof(cl_mem), (void *)&mean_memobj));
  CL_CHECK(clSetKernelArg(std_kernel, 1, sizeof(cl_mem), (void *)&std_memobj));
  CL_CHECK(clSetKernelArg(std_kernel, 2, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(std_kernel, 3, sizeof(float), (void *)&float_n));
  CL_CHECK(clSetKernelArg(std_kernel, 4, sizeof(float), (void *)&eps));
  CL_CHECK(clSetKernelArg(std_kernel, 5, sizeof(int), (void *)&m));
  CL_CHECK(clSetKernelArg(std_kernel, 6, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), (void *)&mean_memobj));
  CL_CHECK(clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), (void *)&std_memobj));
  CL_CHECK(clSetKernelArg(reduce_kernel, 2, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(reduce_kernel, 3, sizeof(float), (void *)&float_n));
  CL_CHECK(clSetKernelArg(reduce_kernel, 4, sizeof(int), (void *)&m));
  CL_CHECK(clSetKernelArg(reduce_kernel, 5, sizeof(int), (void *)&n));

  CL_CHECK(clSetKernelArg(corr_kernel, 0, sizeof(cl_mem), (void *)&symmat_memobj));
  CL_CHECK(clSetKernelArg(corr_kernel, 1, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(corr_kernel, 2, sizeof(int), (void *)&m));
  CL_CHECK(clSetKernelArg(corr_kernel, 3, sizeof(int), (void *)&n));

  // Host-side input/output buffers
  std::vector<float> h_data(data_size);
  std::vector<float> h_mean(vector_size, 0.0f);
  std::vector<float> h_std(vector_size, 0.0f);
  std::vector<float> h_symmat(symmat_size, 0.0f);
  std::vector<float> h_o(symmat_size, 0.0f);

  // Generate input values
  for (size_t i = 0; i < data_size; ++i) {
    h_data[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, data_memobj, CL_TRUE, 0, data_nbytes, h_data.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, symmat_memobj, CL_TRUE, 0, symmat_nbytes, h_symmat.data(), 0, NULL, NULL));

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
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, std_kernel, 1, NULL, &global_1d_m, &local_1d, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, reduce_kernel, 2, NULL, global_2d, local_2d, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, corr_kernel, 1, NULL, &global_1d_m, &local_1d, 0, NULL, NULL));

  // PolyBench's host sets the last diagonal element after the kernel, because
  // corr_kernel only visits j1 < m-1.
  const float one = 1.0f;
  const size_t last_diag_off = ((size_t)m * m - 1) * sizeof(float);
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, symmat_memobj, CL_TRUE, last_diag_off,
                                sizeof(float), &one, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, symmat_memobj, CL_TRUE, 0, symmat_nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  correlation_cpu(h_symmat.data(), h_mean.data(), h_std.data(), h_data.data(),
                  m, n, float_n, eps);
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
