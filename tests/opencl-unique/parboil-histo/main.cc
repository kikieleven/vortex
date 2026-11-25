// Parboil histo — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/parboil/benchmarks/histo/src/opencl_base/main.cpp
//     (input layout, lmemKB/PRESCAN_THREADS selection, launch geometry)
//   .../histo_prescan.cl, histo_intermediates.cl, histo_main.cl,
//   .../histo_final.cl
//
// Parboil reads a packed image, runs the four-kernel histogram chain and writes
// a reference BMP; this port generates a deterministic grayscale-style image,
// runs the same chain and compares the final 8-bit histogram against the exact
// count of pixel values (capped at 255, which is what the chain produces).
//
// Adaptations (see kernel.cl): 16-work-item groups, KB=8, generic prescan
// reduction, flat descriptor/final accesses, A-extension atomics.
//
// Data: 64x64 pixels in [20000,24000), 256x256 output bins. The value band
// spans several 8192-bin ranges so the prescan/local-range path is exercised.
// Counts are integer, so the comparison is exact.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "common.h"

#define KERNEL_PRESCAN "histo_prescan_kernel"
#define KERNEL_INTER "histo_intermediates_kernel"
#define KERNEL_MAIN "histo_main_kernel"
#define KERNEL_FINAL "histo_final_kernel"

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
  fseek(fp, 0, SEEK_SET);

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
cl_kernel inter_kernel = NULL;
cl_kernel main_kernel = NULL;
cl_kernel final_kernel = NULL;
cl_mem input_memobj = NULL;
cl_mem ranges_memobj = NULL;
cl_mem sm_memobj = NULL;
cl_mem subhisto_memobj = NULL;
cl_mem histo_memobj = NULL;
cl_mem overflow_memobj = NULL;
cl_mem final_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (prescan_kernel) clReleaseKernel(prescan_kernel);
  if (inter_kernel) clReleaseKernel(inter_kernel);
  if (main_kernel) clReleaseKernel(main_kernel);
  if (final_kernel) clReleaseKernel(final_kernel);
  if (program) clReleaseProgram(program);
  if (input_memobj) clReleaseMemObject(input_memobj);
  if (ranges_memobj) clReleaseMemObject(ranges_memobj);
  if (sm_memobj) clReleaseMemObject(sm_memobj);
  if (subhisto_memobj) clReleaseMemObject(subhisto_memobj);
  if (histo_memobj) clReleaseMemObject(histo_memobj);
  if (overflow_memobj) clReleaseMemObject(overflow_memobj);
  if (final_memobj) clReleaseMemObject(final_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 64;      // image edge length
int histo_dim = 256; // output histogram edge length (histo_dim^2 bins)

static void show_usage() {
  printf("Usage: [-n image edge] [-g histogram edge] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:g:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'g':
      histo_dim = atoi(optarg);
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

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 1 || histo_dim < 1) {
    printf("Error: invalid sizes\n");
    return -1;
  }

  const unsigned int img_width = size;
  const unsigned int img_height = size;
  const unsigned int histo_width = histo_dim;
  const unsigned int histo_height = histo_dim;
  const unsigned int img_dim = img_width * img_height;
  const unsigned int total_bins = histo_width * histo_height;
  const unsigned int even_width = ((img_width + 1) / 2) * 2;
  const unsigned int padded_height = ((img_height + UNROLL - 1) / UNROLL) * UNROLL;

  printf("Image=%ux%u, histogram=%ux%u, KB=%d, BINS_PER_BLOCK=%d\n",
         img_width, img_height, histo_width, histo_height, KB, BINS_PER_BLOCK);

  // [20000, 24000): several 8192-bin ranges, positive so the prescan range
  // estimate stays well-defined.
  std::vector<unsigned int> h_img(img_dim);
  srand(1);
  for (unsigned int i = 0; i < img_dim; ++i)
    h_img[i] = 20000 + (unsigned int)(rand() % 4000);

  std::vector<unsigned int> h_input((size_t)even_width * padded_height, 0);
  for (unsigned int y = 0; y < img_height; ++y)
    for (unsigned int x = 0; x < img_width; ++x)
      h_input[(size_t)y * even_width + x] = h_img[(size_t)y * img_width + x];

  std::vector<unsigned int> h_ranges = {0xFFFFFFFFu, 0u};
  std::vector<unsigned char> h_sm((size_t)img_dim * 4, 0);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  input_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_input.size() * sizeof(unsigned int), NULL, &_err));
  ranges_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * sizeof(unsigned int), NULL, &_err));
  sm_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_sm.size(), NULL, &_err));

  printf("Create program from kernel source\n");
  size_t kernel_size = 0;
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  prescan_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_PRESCAN, &_err));
  inter_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_INTER, &_err));
  main_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_MAIN, &_err));
  final_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_FINAL, &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, input_memobj, CL_TRUE, 0, h_input.size() * sizeof(unsigned int), h_input.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, ranges_memobj, CL_TRUE, 0, 2 * sizeof(unsigned int), h_ranges.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, sm_memobj, CL_TRUE, 0, h_sm.size(), h_sm.data(), 0, NULL, NULL));

  auto time_start = std::chrono::high_resolution_clock::now();

  printf("Execute prescan kernel\n");
  CL_CHECK(clSetKernelArg(prescan_kernel, 0, sizeof(cl_mem), (void *)&input_memobj));
  CL_CHECK(clSetKernelArg(prescan_kernel, 1, sizeof(unsigned int), (void *)&img_dim));
  CL_CHECK(clSetKernelArg(prescan_kernel, 2, sizeof(cl_mem), (void *)&ranges_memobj));
  {
    const size_t global = (size_t)PRESCAN_GROUPS * BLOCK_SIZE;
    const size_t local = BLOCK_SIZE;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, prescan_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));

  CL_CHECK(clEnqueueReadBuffer(commandQueue, ranges_memobj, CL_TRUE, 0, 2 * sizeof(unsigned int),
                               h_ranges.data(), 0, NULL, NULL));
  unsigned int range_min = h_ranges[0];
  unsigned int range_max = h_ranges[1];
  printf("Range=[%u,%u] (blocks of %d bins)\n", range_min, range_max, BINS_PER_BLOCK);
  if (range_min > range_max || (range_max + 1) * BINS_PER_BLOCK > total_bins) {
    printf("Error: prescan produced an invalid range\n");
    cleanup();
    return -1;
  }
  const unsigned int num_ranges = range_max - range_min + 1;

  printf("Allocate histogram buffers\n");
  const size_t subhisto_len = (size_t)(range_max + 1) * BINS_PER_BLOCK;
  std::vector<unsigned int> h_subhisto(subhisto_len, 0);
  std::vector<unsigned short> h_histo(total_bins, 0);
  std::vector<unsigned int> h_overflow(total_bins, 0);
  std::vector<unsigned char> h_final(total_bins, 0);
  subhisto_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, subhisto_len * sizeof(unsigned int), NULL, &_err));
  histo_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, total_bins * sizeof(unsigned short), NULL, &_err));
  overflow_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, total_bins * sizeof(unsigned int), NULL, &_err));
  final_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, total_bins * sizeof(unsigned char), NULL, &_err));

  CL_CHECK(clEnqueueWriteBuffer(commandQueue, subhisto_memobj, CL_TRUE, 0, subhisto_len * sizeof(unsigned int), h_subhisto.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, histo_memobj, CL_TRUE, 0, total_bins * sizeof(unsigned short), h_histo.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, overflow_memobj, CL_TRUE, 0, total_bins * sizeof(unsigned int), h_overflow.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, final_memobj, CL_TRUE, 0, total_bins * sizeof(unsigned char), h_final.data(), 0, NULL, NULL));

  printf("Execute intermediates kernel\n");
  CL_CHECK(clSetKernelArg(inter_kernel, 0, sizeof(cl_mem), (void *)&input_memobj));
  CL_CHECK(clSetKernelArg(inter_kernel, 1, sizeof(unsigned int), (void *)&img_height));
  CL_CHECK(clSetKernelArg(inter_kernel, 2, sizeof(unsigned int), (void *)&img_width));
  unsigned int half_width = even_width / 2;
  CL_CHECK(clSetKernelArg(inter_kernel, 3, sizeof(unsigned int), (void *)&half_width));
  CL_CHECK(clSetKernelArg(inter_kernel, 4, sizeof(cl_mem), (void *)&sm_memobj));
  {
    const size_t global = (size_t)((img_height + UNROLL - 1) / UNROLL) * BLOCK_SIZE;
    const size_t local = BLOCK_SIZE;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, inter_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL));
  }

  printf("Execute main kernel\n");
  CL_CHECK(clSetKernelArg(main_kernel, 0, sizeof(cl_mem), (void *)&sm_memobj));
  CL_CHECK(clSetKernelArg(main_kernel, 1, sizeof(unsigned int), (void *)&img_dim));
  CL_CHECK(clSetKernelArg(main_kernel, 2, sizeof(unsigned int), (void *)&range_min));
  CL_CHECK(clSetKernelArg(main_kernel, 3, sizeof(unsigned int), (void *)&range_max));
  CL_CHECK(clSetKernelArg(main_kernel, 4, sizeof(unsigned int), (void *)&histo_height));
  CL_CHECK(clSetKernelArg(main_kernel, 5, sizeof(unsigned int), (void *)&histo_width));
  CL_CHECK(clSetKernelArg(main_kernel, 6, sizeof(cl_mem), (void *)&subhisto_memobj));
  CL_CHECK(clSetKernelArg(main_kernel, 7, sizeof(cl_mem), (void *)&histo_memobj));
  CL_CHECK(clSetKernelArg(main_kernel, 8, sizeof(cl_mem), (void *)&overflow_memobj));
  {
    const size_t global[2] = {(size_t)MAIN_GROUPS * BLOCK_SIZE, num_ranges};
    const size_t local[2] = {BLOCK_SIZE, 1};
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, main_kernel, 2, NULL,
                                    global, local, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));

  printf("Execute final kernel\n");
  CL_CHECK(clSetKernelArg(final_kernel, 0, sizeof(unsigned int), (void *)&range_min));
  CL_CHECK(clSetKernelArg(final_kernel, 1, sizeof(unsigned int), (void *)&range_max));
  CL_CHECK(clSetKernelArg(final_kernel, 2, sizeof(unsigned int), (void *)&histo_height));
  CL_CHECK(clSetKernelArg(final_kernel, 3, sizeof(unsigned int), (void *)&histo_width));
  CL_CHECK(clSetKernelArg(final_kernel, 4, sizeof(cl_mem), (void *)&subhisto_memobj));
  CL_CHECK(clSetKernelArg(final_kernel, 5, sizeof(cl_mem), (void *)&histo_memobj));
  CL_CHECK(clSetKernelArg(final_kernel, 6, sizeof(cl_mem), (void *)&overflow_memobj));
  CL_CHECK(clSetKernelArg(final_kernel, 7, sizeof(cl_mem), (void *)&final_memobj));
  {
    const size_t global = (size_t)FINAL_GROUPS * BLOCK_SIZE;
    const size_t local = BLOCK_SIZE;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, final_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, final_memobj, CL_TRUE, 0, total_bins * sizeof(unsigned char),
                               h_final.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Exact histogram of the pixel values, capped at 255 (the chain stores
  // 8-bit counts; the overflow path is not exercised at these counts).
  std::vector<unsigned int> count(total_bins, 0);
  for (unsigned int i = 0; i < img_dim; ++i) {
    if (h_img[i] < total_bins)
      count[h_img[i]]++;
  }

  int errors = 0;
  for (unsigned int i = 0; i < total_bins; ++i) {
    unsigned char expected = (unsigned char)(count[i] < 255 ? count[i] : 255);
    unsigned char actual = h_final[i];
    if (expected != actual) {
      if (errors < 100) {
        printf("*** error: [%u] expected=%u, actual=%u\n", i, expected, actual);
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