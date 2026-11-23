#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <vector>
#include <chrono>
#include "common.h"

// Deterministic LCG in place of the process-wide rand(): PoCL also consumes
// rand(), so separate simx/rtlsim processes would otherwise get different
// input data and their instruction counts could not be compared.
static uint32_t g_lcg_state = 1u;
static uint32_t lcg_rand() {
  g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
  return g_lcg_state >> 1;  // 31 bits: same range as RAND_MAX
}


#define KERNEL_NAME "maxpool2d"

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

static void maxpool2d_cpu(TYPE* out, const TYPE* in,
                          int n, int c, int h, int w,
                          int k, int s, int oh, int ow) {
  for (int nb = 0; nb < n; ++nb)
    for (int ch = 0; ch < c; ++ch)
      for (int oy = 0; oy < oh; ++oy)
        for (int ox = 0; ox < ow; ++ox) {
          float best = -INFINITY;
          for (int ky = 0; ky < k; ++ky) {
            int iy = oy * s + ky;
            if (iy >= h) continue;
            for (int kx = 0; kx < k; ++kx) {
              int ix = ox * s + kx;
              if (ix >= w) continue;
              float v = in[((nb * c + ch) * h + iy) * w + ix];
              if (v > best) best = v;
            }
          }
          out[((nb * c + ch) * oh + oy) * ow + ox] = best;
        }
}

static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue commandQueue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_mem in_memobj = NULL;
static cl_mem out_memobj = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (in_memobj) clReleaseMemObject(in_memobj);
  if (out_memobj) clReleaseMemObject(out_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

static int local_size = 8;

static void show_usage() {
  printf("Usage: [-l local_size] [-h: help]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "l:h")) != -1) {
    switch (c) {
    case 'l':
      local_size = atoi(optarg);
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

  if (local_size <= 0) {
    printf("Error: local_size must be positive\n");
    exit(-1);
  }

  printf("Workload N=%d C=%d H=%d W=%d k=%d s=%d, local_size=%d\n",
         POOL_N, POOL_C, POOL_H, POOL_W, POOL_K, POOL_S, local_size);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  cl_platform_id platform_id;
  size_t kernel_size;

  const int n = POOL_N, c = POOL_C, h = POOL_H, w = POOL_W;
  const int k = POOL_K, s = POOL_S;
  const int oh = (h - k) / s + 1;
  const int ow = (w - k) / s + 1;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  size_t in_count = (size_t)n * c * h * w;
  size_t out_count = (size_t)n * c * oh * ow;
  size_t in_bytes = in_count * sizeof(TYPE);
  size_t out_bytes = out_count * sizeof(TYPE);

  printf("Allocate device buffers\n");
  in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, in_bytes, NULL, &_err));
  out_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_bytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  cl_int build_err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (build_err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 0) {
      std::vector<char> log(log_size + 1);
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
      log[log_size] = 0;
      printf("--- build log ---\n%s\n--- end ---\n", log.data());
    } else {
      printf("(empty build log)\n");
    }
    printf("clBuildProgram failed: %d\n", (int)build_err);
    cleanup();
    exit(-1);
  }

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  std::vector<TYPE> h_in(in_count);
  std::vector<TYPE> h_out(out_count, 0);
  std::vector<TYPE> h_ref(out_count, 0);

  for (size_t i = 0; i < in_count; ++i)
    h_in[i] = (static_cast<float>(lcg_rand()) / RAND_MAX) * 4.0f - 2.0f;

  int nn = n, cc = c, hh = h, ww = w, kk = k, ss = s, ohh = oh, oww = ow;
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void*)&nn));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void*)&cc));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void*)&hh));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void*)&ww));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), (void*)&kk));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), (void*)&ss));
  CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), (void*)&ohh));
  CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), (void*)&oww));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, in_bytes, h_in.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t global_work_size[1] = {(size_t)((out_count + local_size - 1) / local_size * local_size)};
  size_t local_work_size[1] = {(size_t)local_size};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, out_bytes, h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  maxpool2d_cpu(h_ref.data(), h_in.data(), n, c, h, w, k, s, oh, ow);
  int errors = 0;
  for (size_t i = 0; i < out_count; ++i) {
    if (h_out[i] != h_ref[i]) {
      if (errors < 100)
        printf("*** error: [%zu] expected=%f, actual=%f\n", i, h_ref[i], h_out[i]);
      ++errors;
    }
  }
  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors != 0 ? 1 : 0;
}
