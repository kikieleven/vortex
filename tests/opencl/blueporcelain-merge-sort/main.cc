// Blue Porcelain basic/MERGE_SORT — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/MERGE_SORT/merge_sort.cu
//
// Upstream only checks that the final array is non-decreasing; this port also
// compares it element by element against std::sort over the same input, so lost
// or duplicated elements are caught.
//
// Default n is 256 (upstream hard-codes 2048); -n overrides it.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <algorithm>

#define GROUP_SIZE 16

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

// Local deterministic PRNG; see the note in blueporcelain-vectoradd/main.cc.
static uint32_t rng_state = 1;
static uint32_t next_rand() {
  rng_state = rng_state * 1664525u + 1013904223u;
  return rng_state;
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

  printf("Sort size=%d\n", size);

  uint32_t n = (uint32_t)size;
  size_t nbytes = n * sizeof(int64_t);
  size_t local_size = GROUP_SIZE;
  size_t global_size = ((n + local_size - 1) / local_size) * local_size;

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
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

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

  kernel = CL_CHECK2(clCreateKernel(program, "gpu_mergesort", &_err));

  std::vector<int64_t> h_a(n);
  for (uint32_t i = 0; i < n; ++i)
    h_a[i] = (int64_t)(next_rand() % 1024);

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a.data(), 0, NULL, NULL));

  printf("Execute the kernels\n");
  auto time_start = std::chrono::high_resolution_clock::now();

  cl_mem src = a_memobj;
  cl_mem dst = b_memobj;
  long lsize = (long)n;

  for (long width = 2; width < (lsize << 1); width <<= 1) {
    long nthreads = (long)global_size;
    long slices = lsize / (nthreads * width) + 1;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&dst));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int64_t), &lsize));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int64_t), &width));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int64_t), &slices));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));

    cl_mem tmp = src;
    src = dst;
    dst = tmp;
  }

  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<int64_t> h_out(n);
  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, src, CL_TRUE, 0, nbytes, h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<int64_t> ref(h_a);
  std::sort(ref.begin(), ref.end());

  int errors = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (h_out[i] != ref[i]) {
      if (errors < 100)
        printf("*** error: [%d] expected=%lld, actual=%lld\n",
               i, (long long)ref[i], (long long)h_out[i]);
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