// Blue Porcelain basic/PREFIX_SUM — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/PREFIX_SUM/
// (prefix_sum.cu drives scan.cu's sum_scan_blelloch). The host-side structure is
// the same: prescan the data (exclusive scan per block + block totals), scan the
// block totals, then add each block's total to its elements. The block-total
// scan recurses on itself when there are more block totals than one block can
// hold, exactly as upstream.
//
// Adaptations: 16-work-item groups, 32 elements per block (see kernel.cl), and
// the in-place block-sum scan uses a copy as the input argument instead of
// aliasing one buffer as both const and non-const kernel arguments.
//
// Default n is 515 (upstream hard-codes 2^10 + 3); -n overrides it. n > 1024
// exercises the recursive block-sum scan.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <stdint.h>

#define GROUP_SIZE 16
#define MAX_ELEMS_PER_BLOCK (2 * GROUP_SIZE)

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

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel prescan_kernel = NULL;
cl_kernel add_block_sums_kernel = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (prescan_kernel) clReleaseKernel(prescan_kernel);
  if (add_block_sums_kernel) clReleaseKernel(add_block_sums_kernel);
  if (program) clReleaseProgram(program);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// Exclusive scan of `len` elements from d_in into d_out, plus the block-offset
// pass. Recurses on the block totals when they exceed one block.
static void scan_device(uint32_t len, cl_mem d_out, cl_mem d_in) {
  uint32_t grid = (len + MAX_ELEMS_PER_BLOCK - 1) / MAX_ELEMS_PER_BLOCK;
  uint32_t max_elems = MAX_ELEMS_PER_BLOCK;
  size_t bs_nbytes = (size_t)grid * sizeof(uint32_t);
  size_t local_size = GROUP_SIZE;

  cl_mem block_sums = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, bs_nbytes, NULL, &_err));

  size_t global_size = (size_t)grid * GROUP_SIZE;
  CL_CHECK(clSetKernelArg(prescan_kernel, 0, sizeof(cl_mem), (void *)&d_out));
  CL_CHECK(clSetKernelArg(prescan_kernel, 1, sizeof(cl_mem), (void *)&d_in));
  CL_CHECK(clSetKernelArg(prescan_kernel, 2, sizeof(cl_mem), (void *)&block_sums));
  CL_CHECK(clSetKernelArg(prescan_kernel, 3, sizeof(uint32_t), &len));
  CL_CHECK(clSetKernelArg(prescan_kernel, 4, sizeof(uint32_t), &max_elems));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, prescan_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));

  if (grid <= MAX_ELEMS_PER_BLOCK) {
    // One block can scan all block totals. Use a copy as the source so the
    // buffer is never aliased as both a const and a non-const argument.
    cl_mem bs_in = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, bs_nbytes, NULL, &_err));
    CL_CHECK(clEnqueueCopyBuffer(commandQueue, block_sums, bs_in, 0, 0, bs_nbytes, 0, NULL, NULL));
    cl_mem dummy = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &_err));

    uint32_t block_len = grid;
    size_t block_global = GROUP_SIZE;
    CL_CHECK(clSetKernelArg(prescan_kernel, 0, sizeof(cl_mem), (void *)&block_sums));
    CL_CHECK(clSetKernelArg(prescan_kernel, 1, sizeof(cl_mem), (void *)&bs_in));
    CL_CHECK(clSetKernelArg(prescan_kernel, 2, sizeof(cl_mem), (void *)&dummy));
    CL_CHECK(clSetKernelArg(prescan_kernel, 3, sizeof(uint32_t), &block_len));
    CL_CHECK(clSetKernelArg(prescan_kernel, 4, sizeof(uint32_t), &max_elems));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, prescan_kernel, 1, NULL, &block_global, &local_size, 0, NULL, NULL));

    clReleaseMemObject(dummy);
    clReleaseMemObject(bs_in);
  } else {
    cl_mem bs_in = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, bs_nbytes, NULL, &_err));
    CL_CHECK(clEnqueueCopyBuffer(commandQueue, block_sums, bs_in, 0, 0, bs_nbytes, 0, NULL, NULL));
    scan_device(grid, block_sums, bs_in);
    clReleaseMemObject(bs_in);
  }

  CL_CHECK(clSetKernelArg(add_block_sums_kernel, 0, sizeof(cl_mem), (void *)&d_out));
  // Upstream calls this as (d_out, d_out, block_sums, len): the block offset
  // is added to the already-scanned data, not to the raw input.
  CL_CHECK(clSetKernelArg(add_block_sums_kernel, 1, sizeof(cl_mem), (void *)&d_out));
  CL_CHECK(clSetKernelArg(add_block_sums_kernel, 2, sizeof(cl_mem), (void *)&block_sums));
  CL_CHECK(clSetKernelArg(add_block_sums_kernel, 3, sizeof(uint32_t), &len));
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, add_block_sums_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));

  CL_CHECK(clFinish(commandQueue));

  clReleaseMemObject(block_sums);
}

int size = 515;

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

  printf("Scan size=%d\n", size);

  uint32_t n = (uint32_t)size;
  size_t nbytes = n * sizeof(uint32_t);

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
  cl_mem in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  cl_mem out_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &_err));

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

  prescan_kernel = CL_CHECK2(clCreateKernel(program, "gpu_prescan", &_err));
  add_block_sums_kernel = CL_CHECK2(clCreateKernel(program, "gpu_add_block_sums", &_err));

  std::vector<uint32_t> h_in(n);
  for (uint32_t i = 0; i < n; ++i)
    h_in[i] = i;

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, nbytes, h_in.data(), 0, NULL, NULL));

  printf("Execute the kernels\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  scan_device(n, out_memobj, in_memobj);
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<uint32_t> h_out(n, 0);
  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, nbytes, h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  uint32_t run_sum = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (h_out[i] != run_sum) {
      if (errors < 100)
        printf("*** error: [%d] expected=%u, actual=%u\n", i, run_sum, h_out[i]);
      ++errors;
    }
    run_sum += h_in[i];
  }

  if (errors != 0)
    printf("FAILED! - %d errors\n", errors);
  else
    printf("PASSED!\n");

  clReleaseMemObject(in_memobj);
  clReleaseMemObject(out_memobj);
  cleanup();

  return errors;
}