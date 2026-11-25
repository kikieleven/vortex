// Blue Porcelain basic/BITONIC_SORT — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/BITONIC_SORT/bitonic_sort.cu
//
// The upstream kernel sorts each BLOCK_SIZE block independently; this port does
// the same with 16-element blocks (see kernel.cl). The CPU reference is the
// same contract: every 16-element region must be ascending. n does not have to
// be a multiple of 16 -- the last block pads with INT_MAX and only writes the
// real elements.
//
// Default n is 256 (upstream hard-codes 1024); -n overrides it.

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

#define BLOCK_SIZE 16

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
cl_mem values_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (values_memobj) clReleaseMemObject(values_memobj);
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

  printf("Sort size=%d (block size=%d)\n", size, BLOCK_SIZE);

  uint32_t n = (uint32_t)size;
  size_t nbytes = n * sizeof(int32_t);

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
  values_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

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

  kernel = CL_CHECK2(clCreateKernel(program, "bitonic_sort", &_err));

  size_t local_size = BLOCK_SIZE;
  size_t global_size = ((n + local_size - 1) / local_size) * local_size;

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&values_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(uint32_t), &n));

  std::vector<int32_t> h_values(n);
  for (uint32_t i = 0; i < n; ++i)
    h_values[i] = next_rand() % 1000;

  std::vector<int32_t> ref(h_values);
  std::sort(ref.begin(), ref.end());

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, values_memobj, CL_TRUE, 0, nbytes, h_values.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, values_memobj, CL_TRUE, 0, nbytes, h_values.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;

  // Bitonic sorts each 16-element block; n is the full array length and every
  // element must be in its block-local sorted position.
  uint32_t block_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (uint32_t b = 0; b < block_count; ++b) {
    uint32_t start = b * BLOCK_SIZE;
    uint32_t stop = std::min<uint32_t>(start + BLOCK_SIZE, n);
    std::vector<int32_t> block(h_values.begin() + start, h_values.begin() + stop);
    std::sort(block.begin(), block.end());
    for (uint32_t k = 0; k < block.size(); ++k) {
      if (h_values[start + k] != block[k]) {
        if (errors < 100)
          printf("*** error: [%d] expected=%d, actual=%d\n",
                 start + k, block[k], h_values[start + k]);
        ++errors;
      }
    }
  }

  if (errors != 0)
    printf("FAILED! - %d errors\n", errors);
  else
    printf("PASSED!\n");

  cleanup();

  return errors;
}