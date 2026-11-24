// SHOC Sort (level1/sort) — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/shoc/src/opencl/level1/sort/sort.cl
//   third_benchmark/shoc/src/opencl/level1/sort/Sort.cpp:40-62   (verifySort)
//   third_benchmark/shoc/src/opencl/level1/sort/Sort.cpp:150-380 (host)
//
// SHOC uses an LSD radix sort with a 4-bit digit (16 bins) and eight passes
// over 32-bit keys. The pipeline is reduce -> top_scan -> bottom_scan per
// digit; reduce and bottom_scan are non-in-place, so the input/output buffers
// swap each pass. With eight passes the result ends in d_idata, which is where
// Sort.cpp reads it back (Sort.cpp:363).
//
// This is pure integer code (FPTYPE = uint); SHOC has no precision switch for
// this case and none is added.
//
// Verification: SHOC only checks non-decreasing order (verifySort). This port
// compares the full result against std::sort, which also catches lost or
// duplicated keys.
//
// Work-group sizing: SHOC hard-codes 256 and refuses smaller devices
// (Sort.cpp:176-178). Vortex caps a work-group at 16, and top_scan scans the
// per-block histogram in one group, so this port uses LWS=16 and 16 groups.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <algorithm>

#define KERNEL_REDUCE "reduce"
#define KERNEL_TOP_SCAN "top_scan"
#define KERNEL_BOTTOM_SCAN "bottom_scan"

// Vortex's OpenCL device caps a work-group at 16 work-items. The bottom_scan
// kernel initializes its 16 radix bins with `get_local_id(0) < 16`, so LWS must
// be at least 16; top_scan scans the per-block histogram in one group, so the
// number of groups must not exceed LWS.
#define LWS 16
#define NUM_GROUPS 16

#define RADIX_WIDTH 4
#define NUM_DIGITS 16

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

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel reduce_kernel = NULL;
cl_kernel top_scan_kernel = NULL;
cl_kernel bottom_scan_kernel = NULL;
cl_mem in_memobj = NULL;
cl_mem out_memobj = NULL;
cl_mem isums_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (reduce_kernel) clReleaseKernel(reduce_kernel);
  if (top_scan_kernel) clReleaseKernel(top_scan_kernel);
  if (bottom_scan_kernel) clReleaseKernel(bottom_scan_kernel);
  if (program) clReleaseProgram(program);
  if (in_memobj) clReleaseMemObject(in_memobj);
  if (out_memobj) clReleaseMemObject(out_memobj);
  if (isums_memobj) clReleaseMemObject(isums_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 1024;

static void show_usage() {
  printf("Usage: [-n elements] [-h: help]\n");
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

  printf("uint workload size=%d\n", size);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // bottom_scan reads 4-element vectors and reduce/bottom_scan must give each
  // group at least one vector.
  if (size < NUM_GROUPS * 4 || (size % 4) != 0) {
    printf("Error: size must be a positive multiple of 4 and at least %d\n",
           NUM_GROUPS * 4);
    return -1;
  }

  const unsigned int n = (unsigned int)size;
  const size_t nbytes = (size_t)n * sizeof(unsigned int);
  const size_t isums_nbytes = (size_t)NUM_DIGITS * NUM_GROUPS * sizeof(unsigned int);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  printf("Create program from kernel source\n");
  size_t kernel_size = 0;
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  reduce_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_REDUCE, &_err));
  top_scan_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_TOP_SCAN, &_err));
  bottom_scan_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_BOTTOM_SCAN, &_err));

  in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  out_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  isums_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, isums_nbytes, NULL, &_err));

  int num_groups = NUM_GROUPS;
  CL_CHECK(clSetKernelArg(reduce_kernel, 2, sizeof(unsigned int), (void *)&n));
  CL_CHECK(clSetKernelArg(reduce_kernel, 3, LWS * sizeof(unsigned int), NULL));

  CL_CHECK(clSetKernelArg(top_scan_kernel, 0, sizeof(cl_mem), (void *)&isums_memobj));
  CL_CHECK(clSetKernelArg(top_scan_kernel, 1, sizeof(int), (void *)&num_groups));
  CL_CHECK(clSetKernelArg(top_scan_kernel, 2, 2 * LWS * sizeof(unsigned int), NULL));

  CL_CHECK(clSetKernelArg(bottom_scan_kernel, 1, sizeof(cl_mem), (void *)&isums_memobj));
  CL_CHECK(clSetKernelArg(bottom_scan_kernel, 3, sizeof(unsigned int), (void *)&n));
  CL_CHECK(clSetKernelArg(bottom_scan_kernel, 4, 2 * LWS * sizeof(unsigned int), NULL));

  // SHOC Sort.cpp:255: input pattern i % 16.
  std::vector<unsigned int> h_idata(n);
  for (unsigned int i = 0; i < n; ++i)
    h_idata[i] = i % NUM_DIGITS;

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, nbytes,
                                h_idata.data(), 0, NULL, NULL));

  printf("Execute the kernels\n");
  const size_t lws = LWS;
  const size_t group_gws = (size_t)NUM_GROUPS * LWS;
  const size_t top_gws = LWS;

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int shift = 0; shift < (int)(sizeof(unsigned int) * 8); shift += RADIX_WIDTH) {
    const bool even = ((shift / RADIX_WIDTH) % 2 == 0);

    cl_mem pass_in = even ? in_memobj : out_memobj;
    cl_mem pass_out = even ? out_memobj : in_memobj;

    int shift_arg = shift;
    CL_CHECK(clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), (void *)&pass_in));
    CL_CHECK(clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), (void *)&isums_memobj));
    CL_CHECK(clSetKernelArg(reduce_kernel, 4, sizeof(int), (void *)&shift_arg));

    CL_CHECK(clSetKernelArg(bottom_scan_kernel, 0, sizeof(cl_mem), (void *)&pass_in));
    CL_CHECK(clSetKernelArg(bottom_scan_kernel, 2, sizeof(cl_mem), (void *)&pass_out));
    CL_CHECK(clSetKernelArg(bottom_scan_kernel, 5, sizeof(int), (void *)&shift_arg));

    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, reduce_kernel, 1, NULL,
                                    &group_gws, &lws, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, top_scan_kernel, 1, NULL,
                                    &top_gws, &lws, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, bottom_scan_kernel, 1, NULL,
                                    &group_gws, &lws, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Eight passes: pass 7 is odd, so the result is in in_memobj (d_idata), as
  // in SHOC Sort.cpp:363.
  printf("Download destination buffer\n");
  std::vector<unsigned int> h_odata(n);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, in_memobj, CL_TRUE, 0, nbytes,
                               h_odata.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<unsigned int> h_ref(h_idata);
  std::sort(h_ref.begin(), h_ref.end());

  int errors = 0;
  for (unsigned int i = 0; i < n; ++i) {
    if (h_ref[i] != h_odata[i]) {
      if (errors < 100) {
        printf("*** error: [%u] expected=%u, actual=%u\n",
               i, h_ref[i], h_odata[i]);
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
