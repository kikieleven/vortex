// 3D convolution (PolyBench-ACC stencils/convolution-3d) — standalone
// self-checking OpenCL port for Vortex.
//
// The kernel and the CPU reference below evaluate the same 15-term expression
// (see kernel.cl; the upstream stencil is degenerate and is reproduced as-is).
// PolyBench does not write the border, so B is zero-filled on the device before
// launch and the whole n x n x n volume is compared: a stray write outside the
// interior also fails the test.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
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

#define KERNEL_NAME "Convolution3D_kernel"

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

// A[i][j][k] in a volume of shape (ni, nj, nk), row-major.
static inline float at3(const float* A, int plane, int nk, int i, int j, int k) {
  return A[i * plane + j * nk + k];
}

static void conv3d_cpu(float* B, const float* A, int ni, int nj, int nk) {
  float c11, c12, c13, c21, c22, c23, c31, c32, c33;
  c11 = +2;  c21 = +5;  c31 = -8;
  c12 = -3;  c22 = +6;  c32 = -9;
  c13 = +4;  c23 = +7;  c33 = +10;

  const int plane = nk * nj;

  for (int i = 1; i < ni - 1; ++i) {
    for (int j = 1; j < nj - 1; ++j) {
      for (int k = 1; k < nk - 1; ++k) {
        B[i * plane + j * nk + k] =
              c11 * at3(A, plane, nk, i - 1, j - 1, k - 1)
            + c13 * at3(A, plane, nk, i + 1, j - 1, k - 1)
            + c21 * at3(A, plane, nk, i - 1, j - 1, k - 1)
            + c23 * at3(A, plane, nk, i + 1, j - 1, k - 1)
            + c31 * at3(A, plane, nk, i - 1, j - 1, k - 1)
            + c33 * at3(A, plane, nk, i + 1, j - 1, k - 1)
            + c12 * at3(A, plane, nk, i + 0, j - 1, k + 0)
            + c22 * at3(A, plane, nk, i + 0, j + 0, k + 0)
            + c32 * at3(A, plane, nk, i + 0, j + 1, k + 0)
            + c11 * at3(A, plane, nk, i - 1, j - 1, k + 1)
            + c13 * at3(A, plane, nk, i + 1, j - 1, k + 1)
            + c21 * at3(A, plane, nk, i - 1, j + 0, k + 1)
            + c23 * at3(A, plane, nk, i + 1, j + 0, k + 1)
            + c31 * at3(A, plane, nk, i - 1, j + 1, k + 1)
            + c33 * at3(A, plane, nk, i + 1, j + 1, k + 1);
      }
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

  if (size < 3) {
    printf("Error: size must be at least 3\n");
    return -1;
  }

  const int ni = size;
  const int nj = size;
  const int nk = size;
  const size_t volume = (size_t)ni * nj * nk;
  const size_t volume_nbytes = volume * sizeof(float);

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, volume_nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, volume_nbytes, NULL, &_err));

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
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void *)&nk));

  // B starts as zero on both sides so the untouched border can be checked.
  std::vector<float> h_a(volume);
  std::vector<float> h_b(volume, 0.0f);
  std::vector<float> h_o(volume, 0.0f);

  // Generate input values
  for (size_t i = 0; i < volume; ++i) {
    h_a[i] = generate();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, volume_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, volume_nbytes, h_b.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items, so use a 4x4
  // work-group (PolyBench's original 32x8 is not portable here) and round both
  // global dimensions up to a multiple of the local size. The plane index is
  // folded into dimension 1 (see kernel.cl).
  size_t global_work_size[2];
  size_t local_work_size[2] = {4, 4};
  global_work_size[0] = ((nk + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
  size_t planes = (size_t)ni * nj;
  global_work_size[1] = ((planes + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, b_memobj, CL_TRUE, 0, volume_nbytes, h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  conv3d_cpu(h_b.data(), h_a.data(), ni, nj, nk);
  int errors = 0;
  for (size_t i = 0; i < volume; ++i) {
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
