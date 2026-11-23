// Blue Porcelain basic/RADIX_SORT — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/RADIX_SORT/radix_sort.cu
//
// The algorithm is upstream's: 32 bit-passes, each doing a stable partition
// using the down/up index arrays. Upstream's final check only verifies that the
// array is non-decreasing and not all zeros; this port checks the full multiset
// against std::sort, which also catches lost or duplicated elements.
//
// The upstream `(arr[i] >> idx) & 1 == 1` expression parses as
// `(arr[i] >> idx) & (1 == 1)`, i.e. `& 1` -- the intended semantics. The
// explicit `& 1` form is used in the kernel.
//
// Default n is 256 as upstream; -n overrides it.

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
cl_kernel flag_kernel = NULL;
cl_kernel scatter_kernel = NULL;
cl_mem a_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem flag_memobj = NULL;
cl_mem index_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (flag_kernel) clReleaseKernel(flag_kernel);
  if (scatter_kernel) clReleaseKernel(scatter_kernel);
  if (program) clReleaseProgram(program);
  if (a_memobj) clReleaseMemObject(a_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (flag_memobj) clReleaseMemObject(flag_memobj);
  if (index_memobj) clReleaseMemObject(index_memobj);
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
  size_t nbytes = n * sizeof(int32_t);
  size_t local_size = 16;
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
  flag_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  index_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

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

  flag_kernel = CL_CHECK2(clCreateKernel(program, "generate_flag", &_err));
  scatter_kernel = CL_CHECK2(clCreateKernel(program, "scatter", &_err));

  std::vector<int32_t> h_a(n), h_flag(n), h_idown(n), h_iup(n), h_index(n);
  for (uint32_t i = 0; i < n; ++i)
    h_a[i] = (int32_t)(next_rand() & 0x7fffffffu);

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a_memobj, CL_TRUE, 0, nbytes, h_a.data(), 0, NULL, NULL));

  printf("Execute the kernels\n");
  auto time_start = std::chrono::high_resolution_clock::now();

  cl_mem src = a_memobj;
  cl_mem dst = b_memobj;

  for (int bit = 0; bit < 32; ++bit) {
    CL_CHECK(clSetKernelArg(flag_kernel, 0, sizeof(cl_mem), (void *)&src));
    CL_CHECK(clSetKernelArg(flag_kernel, 1, sizeof(cl_mem), (void *)&flag_memobj));
    CL_CHECK(clSetKernelArg(flag_kernel, 2, sizeof(uint32_t), &n));
    CL_CHECK(clSetKernelArg(flag_kernel, 3, sizeof(int), &bit));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, flag_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));

    CL_CHECK(clEnqueueReadBuffer(commandQueue, flag_memobj, CL_TRUE, 0, nbytes, h_flag.data(), 0, NULL, NULL));

    // Host-side index construction, as upstream's generateIDown/generateIUp.
    int val = 0;
    h_idown[0] = 0;
    for (uint32_t i = 1; i < n; ++i) {
      if (h_flag[i - 1] == 0) ++val;
      h_idown[i] = val;
    }
    val = (int)n - 1;
    h_iup[n - 1] = (int)n - 1;
    for (int i = (int)n - 2; i >= 0; --i) {
      if (h_flag[i + 1] == 1) --val;
      h_iup[i] = val;
    }
    for (uint32_t i = 0; i < n; ++i)
      h_index[i] = h_flag[i] ? h_iup[i] : h_idown[i];

    CL_CHECK(clEnqueueWriteBuffer(commandQueue, index_memobj, CL_TRUE, 0, nbytes, h_index.data(), 0, NULL, NULL));

    CL_CHECK(clSetKernelArg(scatter_kernel, 0, sizeof(cl_mem), (void *)&src));
    CL_CHECK(clSetKernelArg(scatter_kernel, 1, sizeof(cl_mem), (void *)&dst));
    CL_CHECK(clSetKernelArg(scatter_kernel, 2, sizeof(cl_mem), (void *)&index_memobj));
    CL_CHECK(clSetKernelArg(scatter_kernel, 3, sizeof(uint32_t), &n));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, scatter_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));

    cl_mem tmp = src;
    src = dst;
    dst = tmp;
  }

  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // After 32 passes the result is in `src` (upstream's A pointer).
  std::vector<int32_t> h_out(n);
  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, src, CL_TRUE, 0, nbytes, h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<int32_t> ref(h_a);
  std::sort(ref.begin(), ref.end());

  int errors = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (h_out[i] != ref[i]) {
      if (errors < 100)
        printf("*** error: [%d] expected=%d, actual=%d\n", i, ref[i], h_out[i]);
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