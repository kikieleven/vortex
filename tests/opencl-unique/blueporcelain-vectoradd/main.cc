// Blue Porcelain basic/VECTORADD — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/VECTORADD/vectoradd.cu
// Keeps the upstream's three data types (float / int / int64) and its exact
// CPU comparison. All three kernels are launched over the same 1D index space.
//
// Adaptation: N default is 256 (upstream hard-codes 1024) and is overridable
// with -n; work-groups are 16 wide and the global size is rounded up, with the
// kernel's `i < n` guard skipping the extra work-items.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <stdint.h>

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
cl_kernel kernel_f = NULL;
cl_kernel kernel_i = NULL;
cl_kernel kernel_l = NULL;
cl_mem fa_memobj = NULL;
cl_mem fb_memobj = NULL;
cl_mem fc_memobj = NULL;
cl_mem ia_memobj = NULL;
cl_mem ib_memobj = NULL;
cl_mem ic_memobj = NULL;
cl_mem la_memobj = NULL;
cl_mem lb_memobj = NULL;
cl_mem lc_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel_f) clReleaseKernel(kernel_f);
  if (kernel_i) clReleaseKernel(kernel_i);
  if (kernel_l) clReleaseKernel(kernel_l);
  if (program) clReleaseProgram(program);
  if (fa_memobj) clReleaseMemObject(fa_memobj);
  if (fb_memobj) clReleaseMemObject(fb_memobj);
  if (fc_memobj) clReleaseMemObject(fc_memobj);
  if (ia_memobj) clReleaseMemObject(ia_memobj);
  if (ib_memobj) clReleaseMemObject(ib_memobj);
  if (ic_memobj) clReleaseMemObject(ic_memobj);
  if (la_memobj) clReleaseMemObject(la_memobj);
  if (lb_memobj) clReleaseMemObject(lb_memobj);
  if (lc_memobj) clReleaseMemObject(lc_memobj);
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

  printf("Vector size=%d\n", size);

  uint32_t n = (uint32_t)size;
  size_t float_nbytes = n * sizeof(float);
  size_t int_nbytes = n * sizeof(int32_t);
  size_t long_nbytes = n * sizeof(int64_t);

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
  fa_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, float_nbytes, NULL, &_err));
  fb_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, float_nbytes, NULL, &_err));
  fc_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, float_nbytes, NULL, &_err));
  ia_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, int_nbytes, NULL, &_err));
  ib_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, int_nbytes, NULL, &_err));
  ic_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, int_nbytes, NULL, &_err));
  la_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, long_nbytes, NULL, &_err));
  lb_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, long_nbytes, NULL, &_err));
  lc_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, long_nbytes, NULL, &_err));

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

  kernel_f = CL_CHECK2(clCreateKernel(program, "vecadd_float", &_err));
  kernel_i = CL_CHECK2(clCreateKernel(program, "vecadd_int", &_err));
  kernel_l = CL_CHECK2(clCreateKernel(program, "vecadd_long", &_err));

  size_t local_size = 16;
  size_t global_size = ((n + local_size - 1) / local_size) * local_size;

  CL_CHECK(clSetKernelArg(kernel_f, 0, sizeof(cl_mem), (void *)&fa_memobj));
  CL_CHECK(clSetKernelArg(kernel_f, 1, sizeof(cl_mem), (void *)&fb_memobj));
  CL_CHECK(clSetKernelArg(kernel_f, 2, sizeof(cl_mem), (void *)&fc_memobj));
  CL_CHECK(clSetKernelArg(kernel_f, 3, sizeof(uint32_t), &n));

  CL_CHECK(clSetKernelArg(kernel_i, 0, sizeof(cl_mem), (void *)&ia_memobj));
  CL_CHECK(clSetKernelArg(kernel_i, 1, sizeof(cl_mem), (void *)&ib_memobj));
  CL_CHECK(clSetKernelArg(kernel_i, 2, sizeof(cl_mem), (void *)&ic_memobj));
  CL_CHECK(clSetKernelArg(kernel_i, 3, sizeof(uint32_t), &n));

  CL_CHECK(clSetKernelArg(kernel_l, 0, sizeof(cl_mem), (void *)&la_memobj));
  CL_CHECK(clSetKernelArg(kernel_l, 1, sizeof(cl_mem), (void *)&lb_memobj));
  CL_CHECK(clSetKernelArg(kernel_l, 2, sizeof(cl_mem), (void *)&lc_memobj));
  CL_CHECK(clSetKernelArg(kernel_l, 3, sizeof(uint32_t), &n));

  std::vector<float> fa(n), fb(n), fc(n);
  std::vector<int32_t> ia(n), ib(n), ic(n);
  std::vector<int64_t> la(n), lb(n), lc(n);

  for (uint32_t i = 0; i < n; ++i) {
    fa[i] = next_float();
    fb[i] = next_float();
    ia[i] = next_rand() % 200000;
    ib[i] = next_rand() % 200000;
    la[i] = (int64_t)INT32_MAX + i;
    lb[i] = (int64_t)INT32_MAX + n - i;
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, fa_memobj, CL_TRUE, 0, float_nbytes, fa.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, fb_memobj, CL_TRUE, 0, float_nbytes, fb.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, ia_memobj, CL_TRUE, 0, int_nbytes, ia.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, ib_memobj, CL_TRUE, 0, int_nbytes, ib.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, la_memobj, CL_TRUE, 0, long_nbytes, la.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, lb_memobj, CL_TRUE, 0, long_nbytes, lb.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel_f, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel_i, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel_l, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffers\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, fc_memobj, CL_TRUE, 0, float_nbytes, fc.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, ic_memobj, CL_TRUE, 0, int_nbytes, ic.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, lc_memobj, CL_TRUE, 0, long_nbytes, lc.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  for (uint32_t i = 0; i < n; ++i) {
    float fc_ref = fa[i] + fb[i];
    if (fc[i] != fc_ref) {
      if (errors < 100)
        printf("*** error: float[%d] expected=%f, actual=%f\n", i, fc_ref, fc[i]);
      ++errors;
    }
    int32_t ic_ref = ia[i] + ib[i];
    if (ic[i] != ic_ref) {
      if (errors < 100)
        printf("*** error: int[%d] expected=%d, actual=%d\n", i, ic_ref, ic[i]);
      ++errors;
    }
    int64_t lc_ref = la[i] + lb[i];
    if (lc[i] != lc_ref) {
      if (errors < 100)
        printf("*** error: long[%d] expected=%lld, actual=%lld\n",
               i, (long long)lc_ref, (long long)lc[i]);
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