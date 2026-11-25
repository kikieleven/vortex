// doitgen (PolyBench-ACC linear-algebra/kernels/doitgen) — standalone
// self-checking OpenCL port for Vortex.
//
//   sum[r][q][p] = sum_s A[r][q][s] * C4[s][p]
//   A[r][q][p]   = sum[r][q][p]
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// All three dimensions are n (see kernel.cl). A is n^3, C4 is n x n, sum is n^3.
//
// The host loops r and launches kernel1 + kernel2 per slice, as PolyBench does.
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

#define KERNEL1_NAME "doitgen_kernel1"
#define KERNEL2_NAME "doitgen_kernel2"

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
static void doitgen_cpu(float* A, const float* C4, float* sum, int n) {
  const int plane = n * n;
  for (int r = 0; r < n; ++r) {
    for (int q = 0; q < n; ++q) {
      for (int p = 0; p < n; ++p) {
        float acc = 0.0f;
        for (int s = 0; s < n; ++s) {
          acc += A[r * plane + q * n + s] * C4[s * n + p];
        }
        sum[r * plane + q * n + p] = acc;
      }
      for (int p = 0; p < n; ++p) {
        A[r * plane + q * n + p] = sum[r * plane + q * n + p];
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
cl_mem c4_memobj = NULL;
cl_mem sum_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (c4_memobj) clReleaseMemObject(c4_memobj);
  if (sum_memobj) clReleaseMemObject(sum_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 12;

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
  const int nr = n, nq = n, np = n;
  const size_t volume = (size_t)nr * nq * np;
  const size_t c4_size = (size_t)np * np;
  const size_t volume_nbytes = volume * sizeof(float);
  const size_t c4_nbytes = c4_size * sizeof(float);

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, volume_nbytes, NULL, &_err));
  c4_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, c4_nbytes, NULL, &_err));
  sum_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, volume_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel1 = CL_CHECK2(clCreateKernel(program, KERNEL1_NAME, &_err));
  kernel2 = CL_CHECK2(clCreateKernel(program, KERNEL2_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel1, 0, sizeof(int), (void *)&nr));
  CL_CHECK(clSetKernelArg(kernel1, 1, sizeof(int), (void *)&nq));
  CL_CHECK(clSetKernelArg(kernel1, 2, sizeof(int), (void *)&np));
  CL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void *)&c4_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void *)&sum_memobj));

  CL_CHECK(clSetKernelArg(kernel2, 0, sizeof(int), (void *)&nr));
  CL_CHECK(clSetKernelArg(kernel2, 1, sizeof(int), (void *)&nq));
  CL_CHECK(clSetKernelArg(kernel2, 2, sizeof(int), (void *)&np));
  CL_CHECK(clSetKernelArg(kernel2, 3, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 4, sizeof(cl_mem), (void *)&c4_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 5, sizeof(cl_mem), (void *)&sum_memobj));

  // Host-side buffers
  std::vector<float> h_a(volume);
  std::vector<float> h_c4(c4_size);
  std::vector<float> h_sum(volume, 0.0f);
  std::vector<float> h_o(volume, 0.0f);

  // Generate input values
  for (size_t i = 0; i < volume; ++i) {
    h_a[i] = generate();
  }
  for (size_t i = 0; i < c4_size; ++i) {
    h_c4[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, volume_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, c4_memobj, CL_TRUE, 0, c4_nbytes, h_c4.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items, so use a 4x4
  // work-group (PolyBench's original 32x8 is not portable here) and round the
  // global size up to a multiple of the local size.
  size_t global_work_size[2];
  size_t local_work_size[2] = {4, 4};
  global_work_size[0] = ((np + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
  global_work_size[1] = ((nq + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < nr; ++r) {
    CL_CHECK(clSetKernelArg(kernel1, 6, sizeof(int), (void *)&r));
    CL_CHECK(clSetKernelArg(kernel2, 6, sizeof(int), (void *)&r));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, a_memobj, CL_TRUE, 0, volume_nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  doitgen_cpu(h_a.data(), h_c4.data(), h_sum.data(), n);
  int errors = 0;
  for (size_t i = 0; i < volume; ++i) {
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
