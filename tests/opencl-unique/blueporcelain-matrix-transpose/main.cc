// Blue Porcelain basic/MATRIX_TRANSPOSE — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/MATRIX_TRANSPOSE/matrix_transpose.cu
//
// Upstream host bug fixed here: its verification loops only over the m x m
// corner of the n x m result (matrix is m x n with m < n upstream), so most of
// the output is unchecked. This port checks every element of the full result.
//
// Adaptation: upstream uses 16x16 blocks (256 work-items); Vortex caps the
// work-group at 16, so the device side uses 4x4 tiles (see kernel.cl). Default
// sizes are 64 x 128 (upstream hard-codes 1024 x 2048); -m/-n override them.

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

int rows = 64;
int cols = 128;

static void show_usage() {
  printf("Usage: [-m rows] [-n cols] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:h")) != -1) {
    switch (c) {
    case 'm':
      rows = atoi(optarg);
      break;
    case 'n':
      cols = atoi(optarg);
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

  printf("Matrix size=%d x %d\n", rows, cols);

  uint32_t m = (uint32_t)rows;
  uint32_t n = (uint32_t)cols;
  size_t a_nbytes = (size_t)m * n * sizeof(int32_t);
  size_t b_nbytes = (size_t)n * m * sizeof(int32_t);

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
  a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, a_nbytes, NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, b_nbytes, NULL, &_err));

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

  kernel = CL_CHECK2(clCreateKernel(program, "parallel_transpose_mem_coalescing", &_err));

  size_t local_size[2] = {4, 4};
  size_t global_size[2] = {
    ((n + local_size[0] - 1) / local_size[0]) * local_size[0],
    ((m + local_size[1] - 1) / local_size[1]) * local_size[1]
  };

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(uint32_t), &m));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(uint32_t), &n));

  std::vector<int32_t> h_a((size_t)m * n);
  std::vector<int32_t> h_b((size_t)n * m, 0);
  for (size_t i = 0; i < h_a.size(); ++i)
    h_a[i] = next_rand() % 100;

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, a_nbytes, h_a.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, b_memobj, CL_TRUE, 0, b_nbytes, h_b.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      int32_t expected = h_a[(size_t)i * n + j];
      int32_t actual = h_b[(size_t)j * m + i];
      if (actual != expected) {
        if (errors < 100)
          printf("*** error: B[%d][%d] expected=%d, actual=%d\n",
                 j, i, expected, actual);
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