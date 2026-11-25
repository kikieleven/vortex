// batchnorm2d (Ventus-OpenCL-Testcase, _get_case/AIops/BatchNorm2d) —
// standalone self-checking OpenCL port for Vortex.
//
//   y[c,y,x] = gamma[c] * (x[c,y,x] - mean[c]) / sqrt(var[c] + eps) + beta[c]
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/BatchNorm2d/
// — the directory has only bn2d.cl, so the host and CPU reference below are
// new. The CPU reference uses the same expression order as the kernel:
// `gamma * (x - mean) * inv_std + beta`, with `inv_std = 1.0f / sqrtf(var+eps)`.
//
// Notes:
//   * the launch keeps the upstream 3D NDRange {C, H, W}; local {1,4,4},
//     global rounded up, kernel guard drops the extra work-items.
//   * inputs come from a private deterministic LCG (porting guide §13.2).
//     `var` is generated in [0.1, 2.0) and `eps` is 1e-5 so the divisor is
//     always positive and well conditioned.
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL: the
//     device may contract `a*b + c` into an FMA, which changes the last bits
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

#define KERNEL_NAME "batchnorm2d_infer"

#define REL_TOL 1e-4f
#define ABS_TOL 1e-5f

#define EPS 1e-5f

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
static float generate_pm1() { return generate() * 2.0f - 1.0f; }  // [-1, 1)

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

// Same expression order as kernel.cl.
static void batchnorm2d_cpu(float* y, const float* x,
                            const float* gamma, const float* beta,
                            const float* mean, const float* var,
                            float eps, int C, int H, int W) {
  const int HW = H * W;
  for (int c = 0; c < C; ++c) {
    const float inv_std = 1.0f / sqrtf(var[c] + eps);
    for (int i = 0; i < HW; ++i) {
      const int idx = c * HW + i;
      y[idx] = gamma[c] * (x[idx] - mean[c]) * inv_std + beta[c];
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
cl_mem x_memobj = NULL;
cl_mem g_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem m_memobj = NULL;
cl_mem v_memobj = NULL;
cl_mem y_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (x_memobj) clReleaseMemObject(x_memobj);
  if (g_memobj) clReleaseMemObject(g_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (m_memobj) clReleaseMemObject(m_memobj);
  if (v_memobj) clReleaseMemObject(v_memobj);
  if (y_memobj) clReleaseMemObject(y_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int dim_c = 4;
int dim_h = 8;
int dim_w = 12;

static void show_usage() {
  printf("Usage: [-c C] [-y H] [-x W] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "c:y:x:h")) != -1) {
    switch (c) {
    case 'c':
      dim_c = atoi(optarg);
      break;
    case 'y':
      dim_h = atoi(optarg);
      break;
    case 'x':
      dim_w = atoi(optarg);
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

  printf("float workload C=%d H=%d W=%d eps=%g\n", dim_c, dim_h, dim_w, EPS);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (dim_c < 1 || dim_h < 1 || dim_w < 1) {
    printf("Error: all dimensions must be at least 1\n");
    return -1;
  }

  const int C = dim_c;
  const int H = dim_h;
  const int W = dim_w;
  const size_t n = (size_t)C * H * W;
  const float eps = EPS;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &_err));
  g_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, C * sizeof(float), NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, C * sizeof(float), NULL, &_err));
  m_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, C * sizeof(float), NULL, &_err));
  v_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, C * sizeof(float), NULL, &_err));
  y_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int arg = 0;
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&g_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&m_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&v_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_float), (void *)&eps));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&y_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&C));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&H));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&W));

  std::vector<float> h_x(n), h_g(C), h_b(C), h_m(C), h_v(C);
  std::vector<float> h_y(n, 0.0f), h_ref(n);
  for (size_t i = 0; i < n; ++i)
    h_x[i] = generate_pm1();
  for (int c = 0; c < C; ++c) {
    h_g[c] = generate_pm1();
    h_b[c] = generate_pm1();
    h_m[c] = generate_pm1();
    h_v[c] = 0.1f + generate() * 1.9f;       // [0.1, 2.0)
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, n * sizeof(float), h_x.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, g_memobj, CL_TRUE, 0, C * sizeof(float), h_g.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, C * sizeof(float), h_b.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, m_memobj, CL_TRUE, 0, C * sizeof(float), h_m.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, v_memobj, CL_TRUE, 0, C * sizeof(float), h_v.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t local_work_size[3] = {1, 4, 4};
  size_t global_work_size[3];
  global_work_size[0] = round_up((size_t)C, local_work_size[0]);
  global_work_size[1] = round_up((size_t)H, local_work_size[1]);
  global_work_size[2] = round_up((size_t)W, local_work_size[2]);
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, y_memobj, CL_TRUE, 0, n * sizeof(float), h_y.data(), 0, NULL, NULL));

  printf("Verify result\n");
  batchnorm2d_cpu(h_ref.data(), h_x.data(), h_g.data(), h_b.data(),
                  h_m.data(), h_v.data(), eps, C, H, W);

  int errors = 0;
  for (size_t i = 0; i < n; ++i) {
    float a = h_y[i];
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