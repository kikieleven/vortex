// softmax (Ventus-OpenCL-Testcase, _get_case/AIops/Softmax) — standalone
// self-checking OpenCL port for Vortex.
//
// Two kernels, both exercised in one run:
//   softmax_rows : softmax over the last dimension of an [M, N] row-major matrix
//   softmax_chw  : softmax over the channel dimension of an [C, H, W] tensor
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/Softmax/
// (kernel.cl is the upstream softmax.cl; the CPU references below reproduce the
// kernel's max/exp/sum order in float).
//
// Differences from the upstream host:
//   * the device's OpenCL C headers declare `exp` but not `expf`, so the kernel
//     keeps the upstream `exp(...)` spelling; `exp` is overloaded on the
//     argument type and the argument is float, so it stays in the working type
//     (the device has no cl_khr_fp64). The CPU reference uses `expf`, the same
//     float function (porting guide §4.6 / §4.8).
//   * inputs come from a private deterministic LCG instead of rand(): PoCL
//     shares the process-wide rand() state, which makes the simx/rtlsim
//     retired-instruction counts differ between runs (porting guide §13.2).
//   * all sizes are command-line parameters with small defaults, and every
//     launch is rounded up to the work-group size (Vortex caps a work-group at
//     16 work-items); the kernels' guards drop the extra work-items.
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL.
//
// Options: -m -n for the row test (default 8 x 64), -c -y -x for the CHW test
// (default 8 x 8 x 8). -h prints usage.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_ROWS_NAME "softmax_rows"
#define KERNEL_CHW_NAME  "softmax_chw"

#define REL_TOL 1e-4f
#define ABS_TOL 1e-6f

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
static float generate() {                     // [0, 1)
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<float>(lcg_state >> 8) / static_cast<float>(1u << 24);
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

// Same passes and accumulation order as softmax_rows.
static void softmax_rows_cpu(float* y, const float* x, int M, int N) {
  for (int r = 0; r < M; ++r) {
    const float* xr = x + (size_t)r * N;
    float* yr = y + (size_t)r * N;
    float m = xr[0];
    for (int j = 1; j < N; ++j)
      m = xr[j] > m ? xr[j] : m;
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
      float e = expf(xr[j] - m);
      yr[j] = e;
      sum += e;
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int j = 0; j < N; ++j)
      yr[j] *= inv;
  }
}

// Same passes and accumulation order as softmax_chw.
static void softmax_chw_cpu(float* y, const float* x, int C, int H, int W) {
  const int HW = H * W;
  for (int yy = 0; yy < H; ++yy) {
    for (int xx = 0; xx < W; ++xx) {
      const int col = yy * W + xx;
      float m = x[col];
      for (int c = 1; c < C; ++c) {
        float v = x[c * HW + col];
        m = v > m ? v : m;
      }
      float sum = 0.0f;
      for (int c = 0; c < C; ++c) {
        float e = expf(x[c * HW + col] - m);
        y[c * HW + col] = e;
        sum += e;
      }
      float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
      for (int c = 0; c < C; ++c)
        y[c * HW + col] *= inv;
    }
  }
}

static size_t round_up(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

static int compare(const char* tag, const std::vector<float>& got,
                   const std::vector<float>& ref) {
  int errors = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    float a = got[i];
    float b = ref[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: [%s][%d] expected=%f, actual=%f (diff=%g)\n",
               tag, (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  return errors;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel k_rows = NULL;
cl_kernel k_chw = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (k_rows) clReleaseKernel(k_rows);
  if (k_chw) clReleaseKernel(k_chw);
  if (program) clReleaseProgram(program);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int rows_m = 8;
int rows_n = 64;
int chw_c = 8;
int chw_h = 8;
int chw_w = 8;

static void show_usage() {
  printf("Usage: [-m rows_M] [-n rows_N] [-c C] [-y H] [-x W] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:c:y:x:h")) != -1) {
    switch (c) {
    case 'm':
      rows_m = atoi(optarg);
      break;
    case 'n':
      rows_n = atoi(optarg);
      break;
    case 'c':
      chw_c = atoi(optarg);
      break;
    case 'y':
      chw_h = atoi(optarg);
      break;
    case 'x':
      chw_w = atoi(optarg);
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

  printf("float workload rows=%dx%d chw=%dx%dx%d\n",
         rows_m, rows_n, chw_c, chw_h, chw_w);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (rows_m < 1 || rows_n < 1 || chw_c < 1 || chw_h < 1 || chw_w < 1) {
    printf("Error: all dimensions must be at least 1\n");
    return -1;
  }

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  k_rows = CL_CHECK2(clCreateKernel(program, KERNEL_ROWS_NAME, &_err));
  k_chw  = CL_CHECK2(clCreateKernel(program, KERNEL_CHW_NAME,  &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  int errors = 0;

  // ---- A) row softmax ------------------------------------------------
  {
    const size_t n = (size_t)rows_m * rows_n;

    std::vector<float> h_x(n), h_y(n, 0.0f), h_ref(n);
    for (size_t i = 0; i < n; ++i)
      h_x[i] = generate() * 10.0f - 5.0f;      // upstream frand()*5 in [-5, 5)

    printf("Allocate device buffers (softmax_rows)\n");
    cl_mem x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &_err));
    cl_mem y_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &_err));

    int arg = 0;
    CL_CHECK(clSetKernelArg(k_rows, arg++, sizeof(cl_mem), (void *)&x_memobj));
    CL_CHECK(clSetKernelArg(k_rows, arg++, sizeof(cl_mem), (void *)&y_memobj));
    CL_CHECK(clSetKernelArg(k_rows, arg++, sizeof(int), (void *)&rows_m));
    CL_CHECK(clSetKernelArg(k_rows, arg++, sizeof(int), (void *)&rows_n));

    printf("Upload source buffers\n");
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, n * sizeof(float), h_x.data(), 0, NULL, NULL));

    printf("Execute the kernel (softmax_rows)\n");
    size_t local_work_size[1] = {16};
    size_t global_work_size[1];
    global_work_size[0] = round_up((size_t)rows_m, local_work_size[0]);
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, k_rows, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

    printf("Download destination buffer\n");
    CL_CHECK(clEnqueueReadBuffer(commandQueue, y_memobj, CL_TRUE, 0, n * sizeof(float), h_y.data(), 0, NULL, NULL));

    printf("Verify result (softmax_rows)\n");
    softmax_rows_cpu(h_ref.data(), h_x.data(), rows_m, rows_n);
    int err = compare("rows", h_y, h_ref);
    if (err == 0) printf("softmax_rows: PASSED!\n");
    else          printf("softmax_rows: FAILED! - %d errors\n", err);
    errors += err;

    clReleaseMemObject(x_memobj);
    clReleaseMemObject(y_memobj);
  }

  // ---- B) channel softmax --------------------------------------------
  {
    const size_t n = (size_t)chw_c * chw_h * chw_w;

    std::vector<float> h_x(n), h_y(n, 0.0f), h_ref(n);
    for (size_t i = 0; i < n; ++i)
      h_x[i] = generate() * 10.0f - 5.0f;      // upstream frand()*5 in [-5, 5)

    printf("Allocate device buffers (softmax_chw)\n");
    cl_mem x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &_err));
    cl_mem y_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &_err));

    int arg = 0;
    CL_CHECK(clSetKernelArg(k_chw, arg++, sizeof(cl_mem), (void *)&x_memobj));
    CL_CHECK(clSetKernelArg(k_chw, arg++, sizeof(cl_mem), (void *)&y_memobj));
    CL_CHECK(clSetKernelArg(k_chw, arg++, sizeof(int), (void *)&chw_c));
    CL_CHECK(clSetKernelArg(k_chw, arg++, sizeof(int), (void *)&chw_h));
    CL_CHECK(clSetKernelArg(k_chw, arg++, sizeof(int), (void *)&chw_w));

    printf("Upload source buffers\n");
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, n * sizeof(float), h_x.data(), 0, NULL, NULL));

    printf("Execute the kernel (softmax_chw)\n");
    // 2D NDRange {H, W}; 4x4 work-group, global rounded up.
    size_t local_work_size[2] = {4, 4};
    size_t global_work_size[2];
    global_work_size[0] = round_up((size_t)chw_h, local_work_size[0]);
    global_work_size[1] = round_up((size_t)chw_w, local_work_size[1]);
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, k_chw, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

    printf("Download destination buffer\n");
    CL_CHECK(clEnqueueReadBuffer(commandQueue, y_memobj, CL_TRUE, 0, n * sizeof(float), h_y.data(), 0, NULL, NULL));

    printf("Verify result (softmax_chw)\n");
    softmax_chw_cpu(h_ref.data(), h_x.data(), chw_c, chw_h, chw_w);
    int err = compare("chw", h_y, h_ref);
    if (err == 0) printf("softmax_chw: PASSED!\n");
    else          printf("softmax_chw: FAILED! - %d errors\n", err);
    errors += err;

    clReleaseMemObject(x_memobj);
    clReleaseMemObject(y_memobj);
  }

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}