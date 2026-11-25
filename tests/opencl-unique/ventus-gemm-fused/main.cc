// gemm_relu (Ventus-OpenCL-Testcase, _get_case/AIops/gemm_fused) — standalone
// self-checking OpenCL port for Vortex.
//
//   C[m,n] = max(0, sum_k A'[m,k] * B'[k,n])
//   A' = do_trans_a ? A^T : A      B' = do_trans_b ? B^T : B
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/gemm_fused/
// — the directory has only gemm_fused.cl, so the host and CPU reference below
// are new.
//
// **Upstream defect corrected in kernel.cl.** The transposed index expressions
// upstream wrote use the wrong strides (`k*K + m` for A, `n*N + k` for B) and
// are only correct when the relevant dimensions are equal; with N > K the B
// index reads out of bounds. The kernel uses the correct strides (`k*M + m`,
// `n*K + k`); the non-transposed paths are unchanged. See the kernel header.
//
// Notes:
//   * a bounds guard was added to the kernel because Vortex caps a work-group
//     at 16 work-items, so the global size is rounded up to the 4x4 work-group.
//   * inputs come from a private deterministic LCG in [-1,1) (porting guide
//     §13.2). Signed inputs make the fused ReLU fire; with non-negative inputs
//     every sum would pass through unchanged.
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL: the
//     ReLU makes half the outputs zero and the rest signed, so ULP is unusable
//     (porting guide §4.7).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "gemm_relu"

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

// Deterministic LCG (porting guide §13.2).
static uint32_t lcg_state = 1u;
static float generate_pm1() {                 // [-1, 1)
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<float>(lcg_state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
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

// Same accumulation order and same transposed strides as kernel.cl.
static void gemm_relu_cpu(float* C, const float* A, const float* B,
                          int M, int K, int N, int trans_a, int trans_b) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        int a_idx = (trans_a) ? k * M + i : i * K + k;
        int b_idx = (trans_b) ? j * K + k : k * N + j;
        sum += A[a_idx] * B[b_idx];
      }
      C[i * N + j] = fmaxf(0.0f, sum);
    }
  }
}

static size_t round_up(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
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

int size = 32;   // M = N
int kdim = 16;   // K
int trans_a = 0;
int trans_b = 0;

static void show_usage() {
  printf("Usage: [-n size(M=N)] [-k dim(K)] [-a trans_a(0|1)] [-b trans_b(0|1)] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:a:b:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'k':
      kdim = atoi(optarg);
      break;
    case 'a':
      trans_a = atoi(optarg);
      break;
    case 'b':
      trans_b = atoi(optarg);
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

  printf("float workload M=N=%d K=%d trans_a=%d trans_b=%d\n",
         size, kdim, trans_a, trans_b);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 1 || kdim < 1) {
    printf("Error: sizes must be at least 1\n");
    return -1;
  }

  const int M = size;
  const int N = size;
  const int K = kdim;

  const size_t a_size = (size_t)(trans_a ? K * M : M * K);
  const size_t b_size = (size_t)(trans_b ? N * K : K * N);
  const size_t c_size = (size_t)M * N;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, a_size * sizeof(float), NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, b_size * sizeof(float), NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_size * sizeof(float), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int arg = 0;
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&c_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&M));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&K));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&N));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&trans_a));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&trans_b));

  std::vector<float> h_a(a_size), h_b(b_size);
  std::vector<float> h_o(c_size, 0.0f), h_ref(c_size);
  for (size_t i = 0; i < a_size; ++i)
    h_a[i] = generate_pm1();
  for (size_t i = 0; i < b_size; ++i)
    h_b[i] = generate_pm1();

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, a_size * sizeof(float), h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, b_size * sizeof(float), h_b.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t local_work_size[2] = {4, 4};
  size_t global_work_size[2];
  global_work_size[0] = round_up((size_t)M, local_work_size[0]);
  global_work_size[1] = round_up((size_t)N, local_work_size[1]);
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, c_memobj, CL_TRUE, 0, c_size * sizeof(float), h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  gemm_relu_cpu(h_ref.data(), h_a.data(), h_b.data(), M, K, N, trans_a, trans_b);

  int errors = 0;
  for (size_t i = 0; i < c_size; ++i) {
    float a = h_o[i];
    float b = h_ref[i];
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