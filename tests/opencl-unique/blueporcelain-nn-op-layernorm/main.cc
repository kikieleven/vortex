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


#define KERNEL_NAME "layernorm"

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

static void layernorm_cpu(TYPE* out, const TYPE* in, const TYPE* w, const TYPE* b,
                          int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    int base = r * cols;
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) sum += in[base + j];
    float mean = sum / (float)cols;

    float var = 0.0f;
    for (int j = 0; j < cols; ++j) {
      float d = in[base + j] - mean;
      var += d * d;
    }
    var /= (float)cols;

    float inv = 1.0f / sqrtf(var + eps);
    for (int j = 0; j < cols; ++j)
      out[base + j] = (in[base + j] - mean) * inv * w[j] + b[j];
  }
}

static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue commandQueue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_mem in_memobj = NULL;
static cl_mem w_memobj = NULL;
static cl_mem b_memobj = NULL;
static cl_mem out_memobj = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (in_memobj) clReleaseMemObject(in_memobj);
  if (w_memobj) clReleaseMemObject(w_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (out_memobj) clReleaseMemObject(out_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

static int local_size = 4;

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

  printf("Workload rows=%d cols=%d, local_size=%d\n", LN_ROWS, LN_COLS, local_size);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  cl_platform_id platform_id;
  size_t kernel_size;

  const int rows = LN_ROWS, cols = LN_COLS;
  const float eps = LN_EPS;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  size_t count = (size_t)rows * cols;
  size_t nbytes = count * sizeof(TYPE);
  size_t cbytes = (size_t)cols * sizeof(TYPE);

  printf("Allocate device buffers\n");
  in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  w_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, cbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, cbytes, NULL, &_err));
  out_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

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

  std::vector<TYPE> h_in(count);
  std::vector<TYPE> h_w(cols);
  std::vector<TYPE> h_b(cols);
  std::vector<TYPE> h_out(count, 0);
  std::vector<TYPE> h_ref(count, 0);

  for (size_t i = 0; i < count; ++i)
    h_in[i] = (static_cast<float>(lcg_rand()) / RAND_MAX) * 4.0f - 2.0f;
  for (int j = 0; j < cols; ++j) {
    h_w[j] = (static_cast<float>(lcg_rand()) / RAND_MAX) * 1.5f + 0.5f;
    h_b[j] = (static_cast<float>(lcg_rand()) / RAND_MAX) * 0.5f - 0.25f;
  }

  int ccols = cols;
  float ceps = eps;
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&w_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&out_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void*)&ccols));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(TYPE), (void*)&ceps));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, nbytes, h_in.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_memobj, CL_TRUE, 0, cbytes, h_w.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, cbytes, h_b.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t global_work_size[1] = {(size_t)((rows + local_size - 1) / local_size * local_size)};
  size_t local_work_size[1] = {(size_t)local_size};
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, nbytes, h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  layernorm_cpu(h_ref.data(), h_in.data(), h_w.data(), h_b.data(), rows, cols, eps);
  int errors = 0;
  for (size_t i = 0; i < count; ++i) {
    float diff = fabsf(h_out[i] - h_ref[i]);
    if (diff > 1e-5f + 1e-4f * fabsf(h_ref[i])) {
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
