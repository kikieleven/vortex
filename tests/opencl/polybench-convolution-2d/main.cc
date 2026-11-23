// 2D convolution (PolyBench-ACC stencils/convolution-2d) — standalone
// self-checking OpenCL port for Vortex.
//
// B[i][j] is the 3x3 weighted sum of A centred at (i, j), with PolyBench's
// original coefficients. Kernel and CPU reference evaluate the same expression.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`; the
// fractional coefficients cannot survive an integer type, so this port stays
// single-type like the native conv3 case.
//
// PolyBench semantics: the kernel does not write the border. B is zero-filled
// on the device before launch, and the checker compares the whole n x n matrix,
// so a stray write outside the interior also fails the test.
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the same
// mixed tolerance form used by tests/opencl/{hotspot3D,cfd,lud}. The result is
// signed (the stencil has negative weights) and can pass through zero, so the
// pure-ULP comparator used by conv3/vecadd is not applicable.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "Convolution2D_kernel"

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

// PolyBench coefficients, unchanged (identical to the kernel's).
static void convolution2d_cpu(float *B, const float *A, int ni, int nj) {
  const float c11 = +0.2f;
  const float c21 = +0.5f;
  const float c31 = -0.8f;
  const float c12 = -0.3f;
  const float c22 = +0.6f;
  const float c32 = -0.9f;
  const float c13 = +0.4f;
  const float c23 = +0.7f;
  const float c33 = +0.1f;

  for (int i = 1; i < ni - 1; ++i) {
    for (int j = 1; j < nj - 1; ++j) {
      B[i * nj + j] =
          c11 * A[(i - 1) * nj + (j - 1)] +
          c21 * A[(i - 1) * nj + (j + 0)] +
          c31 * A[(i - 1) * nj + (j + 1)] +
          c12 * A[(i + 0) * nj + (j - 1)] +
          c22 * A[(i + 0) * nj + (j + 0)] +
          c32 * A[(i + 0) * nj + (j + 1)] +
          c13 * A[(i + 1) * nj + (j - 1)] +
          c23 * A[(i + 1) * nj + (j + 0)] +
          c33 * A[(i + 1) * nj + (j + 1)];
    }
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
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

  if (size < 3) {
    printf("Error: size must be at least 3\n");
    return -1;
  }

  const int ni = size;
  const int nj = size;
  const size_t points = (size_t)size * size;
  const size_t matrix_nbytes = points * sizeof(float);

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

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void *)&ni));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void *)&nj));

  // B starts as zero on both sides so the untouched border can be checked.
  std::vector<float> h_a(points);
  std::vector<float> h_b(points, 0.0f);
  std::vector<float> h_o(points, 0.0f);

  // Generate input values
  for (size_t i = 0; i < points; ++i) {
    h_a[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, matrix_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, matrix_nbytes, h_b.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items, so use a 4x4
  // work-group (PolyBench's original 32x8 is not portable here) and round the
  // global size up to a multiple of the local size.
  size_t global_work_size[2];
  size_t local_work_size[2] = {4, 4};
  global_work_size[0] = ((size + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
  global_work_size[1] = ((size + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, b_memobj, CL_TRUE, 0, matrix_nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  convolution2d_cpu(h_b.data(), h_a.data(), ni, nj);
  int errors = 0;
  for (size_t i = 0; i < points; ++i) {
    float a = h_o[i];
    float b = h_b[i];
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
