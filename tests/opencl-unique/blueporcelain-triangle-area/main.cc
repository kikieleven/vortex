// Blue Porcelain basic/TRIANGLE_AREA — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/TRIANGLE_AREA/triangle_area.cu
//
// Upstream host bugs fixed here (guide section 4.1):
//   - `host_triangle_area()` sets `c = arr2[idx]` although the kernel reads
//     `c = arr3[idx]`; the CPU reference here uses arr3, matching the DUT.
//   - its comparison is `abs((cpu_res - res) / cpu_res) > 1e-3`, where `abs()`
//     truncates the float to int and `cpu_res` can be 0 for degenerate
//     triangles. This port uses a float abs+rel test.
//
// Adaptation: N default is 256 (upstream hard-codes 2048), override with -n;
// 16-wide work-groups and an `idx < n` guard.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <math.h>

#define ABS_TOL 1e-5f
#define REL_TOL 1e-5f

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

  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

static bool compare_equal(float a, float b) {
  float tol = ABS_TOL + REL_TOL * fabsf(b);
  return fabsf(a - b) <= tol;
}

static void triangle_area_cpu(const std::vector<float>& a,
                              const std::vector<float>& b,
                              const std::vector<float>& c,
                              std::vector<float>& res,
                              uint32_t n) {
  for (uint32_t idx = 0; idx < n; ++idx) {
    if (a[idx] + b[idx] > c[idx] &&
        a[idx] + c[idx] > b[idx] &&
        b[idx] + c[idx] > a[idx]) {
      float p = (a[idx] + b[idx] + c[idx]) * 0.5f;
      res[idx] = sqrtf(p * (p - a[idx]) * (p - b[idx]) * (p - c[idx]));
    } else {
      res[idx] = 0.0f;
    }
  }
}

// Local deterministic PRNG. Global rand() is shared with PoCL, whose internal
// calls perturb the sequence between runs (observed as a changing input
// checksum and a mismatched simx/rtlsim instruction count).
static uint32_t rng_state = 1;
static uint32_t next_rand() {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
}
static float next_float() {
  return (next_rand() >> 8) * (1.0f / 16777216.0f);
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem c_memobj = NULL;
cl_mem res_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (c_memobj) clReleaseMemObject(c_memobj);
  if (res_memobj) clReleaseMemObject(res_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

int size = 256;

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
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  printf("Triangle count=%d\n", size);

  uint32_t n = (uint32_t)size;
  size_t nbytes = n * sizeof(float);

  cl_platform_id platform_id;
  size_t kernel_size;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  c_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  res_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, "gpu_triangle_area", &_err));

  size_t local_size = 16;
  size_t global_size = ((n + local_size - 1) / local_size) * local_size;

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&res_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(uint32_t), &n));

  std::vector<float> h_a(n), h_b(n), h_c(n), h_res(n, 0.0f);
  for (uint32_t i = 0; i < n; ++i) {
    h_a[i] = next_float();
    h_b[i] = next_float();
    h_c[i] = next_float();
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, nbytes, h_b.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, c_memobj, CL_TRUE, 0, nbytes, h_c.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, res_memobj, CL_TRUE, 0, nbytes, h_res.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<float> ref(n);
  triangle_area_cpu(h_a, h_b, h_c, ref, n);

  int errors = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (!compare_equal(h_res[i], ref[i])) {
      if (errors < 100)
        printf("*** error: [%d] expected=%f, actual=%f\n", i, ref[i], h_res[i]);
      ++errors;
    }
  }

  if (errors != 0)
    printf("FAILED! - %d errors\n", errors);
  else
    printf("PASSED!\n");

  cleanup();

  return errors;
}