// SHOC Reduction (level1/reduction) — standalone self-checking OpenCL port for
// Vortex, single precision only.
//
// Source:
//   third_benchmark/shoc/src/opencl/level1/reduction/reduction.cl
//   third_benchmark/shoc/src/opencl/level1/reduction/Reduction.cpp:40-69
//     (checkResults: sum the per-group results, compare with the host sum)
//   third_benchmark/shoc/src/opencl/level1/reduction/Reduction.cpp:208-330
//     (problem size, per-group layout, launch)
//
// SHOC's OpenCL reference is not host code but the `reduceNoLocal` kernel run
// on the device (Reduction.cpp:198). This port uses the host sum directly,
// which is what SHOC's checkResults compares against.
//
// Data is `i % 3` as in SHOC (Reduction.cpp:227). With this data every partial
// sum is an integer below 2^24, so the float additions are exact regardless of
// order; the comparison is therefore exact (the 1e-8 threshold from
// checkResults is kept as a guard, but the difference should be 0).
//
// Each work-group consumes 2*local_size elements per stride. The host picks the
// number of groups so that every thread's (i, i+local_size) pair exists, which
// the kernel assumes (no bounds guard on i+blockSize).

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "reduce"

// Vortex's OpenCL device caps a work-group at 16 work-items.
#define LWS 16

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
cl_kernel kernel = NULL;
cl_mem input_memobj = NULL;
cl_mem output_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (input_memobj) clReleaseMemObject(input_memobj);
  if (output_memobj) clReleaseMemObject(output_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 2048;

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

  printf("float workload size=%d\n", size);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // Every work-group consumes 2*LWS elements per stride and the kernel reads
  // i+blockSize without a separate bound check, so n must be a multiple of
  // 2*LWS. Pick one stride per thread: num_groups = n / (2*LWS).
  if (size < 2 * LWS || (size % (2 * LWS)) != 0) {
    printf("Error: size must be a positive multiple of %d\n", 2 * LWS);
    return -1;
  }

  const unsigned int n = (unsigned int)size;
  const unsigned int num_groups = n / (2 * LWS);
  const size_t nbytes = (size_t)n * sizeof(float);
  const size_t out_nbytes = (size_t)num_groups * sizeof(float);

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

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  input_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &_err));
  output_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, out_nbytes, NULL, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, LWS * sizeof(float), NULL));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&n));

  // SHOC Reduction.cpp:227.
  std::vector<float> h_idata(n);
  for (unsigned int i = 0; i < n; ++i) {
    h_idata[i] = (float)(i % 3);
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, input_memobj, CL_TRUE, 0, nbytes,
                                h_idata.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t local_work_size = LWS;
  const size_t global_work_size = (size_t)num_groups * LWS;
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  std::vector<float> h_odata(num_groups);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, output_memobj, CL_TRUE, 0, out_nbytes,
                               h_odata.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // SHOC checkResults (Reduction.cpp:40-69).
  float ref_sum = 0.0f;
  for (unsigned int i = 0; i < n; ++i)
    ref_sum += h_idata[i];

  float dev_sum = 0.0f;
  for (unsigned int i = 0; i < num_groups; ++i)
    dev_sum += h_odata[i];

  const float threshold = 1.0e-8f;
  int errors = 0;
  if (fabsf(dev_sum - ref_sum) > threshold) {
    printf("*** error: expected=%f, actual=%f (diff=%g)\n",
           ref_sum, dev_sum, fabsf(dev_sum - ref_sum));
    ++errors;
  }

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}
