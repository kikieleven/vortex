// Parboil sgemm — standalone self-checking OpenCL port for Vortex, single
// precision only.
//
// Source:
//   third_benchmark/parboil/benchmarks/sgemm/src/opencl_base/kernel.cl
//   third_benchmark/parboil/benchmarks/sgemm/src/opencl_base/main.cc
//     (basicSgemm: alpha=1.0f, beta=0.0f, column-major A, B stored transposed)
//
// Parboil has no CPU reference for sgemm (it writes the result to a file); this
// port adds a host reference and an abs/rel comparison.
//
// Adaptation: upstream launches 16x16 = 256-work-item groups (TILE_SZ 16).
// Vortex caps a work-group at 16 work-items, so the kernel is launched with
// 4x4 groups and the global size is rounded up to a multiple of 4. The kernel
// gained a bounds guard for the extra work-items; its math is unchanged.
//
// Data: deterministic rand()/RAND_MAX matrices in [0,1), C starts at zero,
// alpha = 1 (third_benchmark/.../main.cc:180). Both sides accumulate over k
// in ascending order, so only FMA contraction can perturb the result; the
// tolerance is 1e-3 relative + 1e-3 absolute.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "mysgemmNT"

#define LWS 4
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
}

static size_t round_up(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 1) {
    printf("Error: size must be positive\n");
    return -1;
  }

  printf("Matrix size=%d\n", size);

  const int n = size;
  const int lda = n;          // column-major A: A[m + i*lda]
  const int ldb = n;          // B is stored transposed: B[n + i*ldb]
  const int ldc = n;          // column-major C: C[m + n*ldc]
  const float alpha = 1.0f;   // main.cc:180
  const float beta = 0.0f;    // main.cc:180

  const size_t matrix_size = (size_t)n * n;
  const size_t matrix_nbytes = matrix_size * sizeof(float);

  std::vector<float> h_a(matrix_size), h_b(matrix_size);
  srand(1);
  for (size_t i = 0; i < matrix_size; ++i)
    h_a[i] = static_cast<float>(rand()) / RAND_MAX;
  for (size_t i = 0; i < matrix_size; ++i)
    h_b[i] = static_cast<float>(rand()) / RAND_MAX;

  std::vector<float> h_c(matrix_size, 0.0f);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_nbytes, NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  size_t kernel_size = 0;
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), (void *)&lda));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void *)&ldb));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&c_memobj));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void *)&ldc));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), (void *)&n));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(float), (void *)&alpha));
  CL_CHECK(clSetKernelArg(kernel, 8, sizeof(float), (void *)&beta));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, matrix_nbytes, h_b.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, c_memobj, CL_TRUE, 0, matrix_nbytes, h_c.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t global_work_size[2] = {
    round_up((size_t)n, LWS), round_up((size_t)n, LWS)
  };
  const size_t local_work_size[2] = {LWS, LWS};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                  global_work_size, local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, c_memobj, CL_TRUE, 0, matrix_nbytes,
                               h_c.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Host reference: same NT product and the same alpha/beta epilogue, with C
  // starting at zero (main.cc:171-181).
  std::vector<float> h_ref(matrix_size);
  for (int m = 0; m < n; ++m) {
    for (int j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < n; ++i)
        acc += h_a[m + i * lda] * h_b[j + i * ldb];
      h_ref[m + j * ldc] = h_c[m + j * ldc] * beta + alpha * acc;
    }
  }

  int errors = 0;
  for (size_t i = 0; i < matrix_size; ++i) {
    float expected = h_ref[i];
    float actual = h_c[i];
    float tol = REL_TOL * fabsf(expected) + ABS_TOL;
    if (fabsf(actual - expected) > tol) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, expected, actual, fabsf(actual - expected));
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