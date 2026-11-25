// Blue Porcelain basic/NBODY — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/NBODY/n_body.cu
//
// Positions follow upstream: body i sits at (i+1, i+1, i+1), so the influence of
// body j on body i is 1 / (sqrt(3) * |i-j|). The CPU reference is upstream's
// serial double loop. Upstream's relative tolerance (1e-3) is kept and extended
// with an absolute floor, since the upstream test divides by the device result.
//
// Default n is 64 (upstream hard-codes 256); -n overrides it.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <math.h>

#define GROUP_SIZE 16
#define ABS_TOL 1e-4f
#define REL_TOL 1e-3f

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
cl_kernel kernel = NULL;
cl_mem pos_memobj = NULL;
cl_mem force_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (pos_memobj) clReleaseMemObject(pos_memobj);
  if (force_memobj) clReleaseMemObject(force_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

int size = 64;

static void show_usage() {
  printf("Usage: [-n bodies] [-h: help]\n");
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

  printf("Body count=%d\n", size);

  uint32_t n = (uint32_t)size;
  uint32_t bodies_per_thread = (n + GROUP_SIZE - 1) / GROUP_SIZE;
  if (bodies_per_thread == 0) bodies_per_thread = 1;
  size_t pos_nbytes = n * sizeof(cl_float4);
  size_t force_nbytes = n * sizeof(float);

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
  pos_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, pos_nbytes, NULL, &_err));
  force_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, force_nbytes, NULL, &_err));

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

  kernel = CL_CHECK2(clCreateKernel(program, "force_comp", &_err));

  size_t local_size = GROUP_SIZE;
  size_t global_size = n * GROUP_SIZE;

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&pos_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(uint32_t), &n));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&force_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(uint32_t), &bodies_per_thread));

  std::vector<cl_float4> h_pos(n);
  for (uint32_t i = 0; i < n; ++i) {
    float v = (float)(i + 1);
    h_pos[i].s[0] = v;
    h_pos[i].s[1] = v;
    h_pos[i].s[2] = v;
    h_pos[i].s[3] = 0.0f;
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffer\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, pos_memobj, CL_TRUE, 0, pos_nbytes, h_pos.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<float> h_force(n, 0.0f);
  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, force_memobj, CL_TRUE, 0, force_nbytes, h_force.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  for (uint32_t i = 0; i < n; ++i) {
    float ref = 0.0f;
    for (uint32_t j = 0; j < n; ++j) {
      if (j == i) continue;
      float dx = h_pos[i].s[0] - h_pos[j].s[0];
      float dy = h_pos[i].s[1] - h_pos[j].s[1];
      float dz = h_pos[i].s[2] - h_pos[j].s[2];
      ref += 1.0f / sqrtf(dx * dx + dy * dy + dz * dz);
    }
    float tol = ABS_TOL + REL_TOL * fabsf(ref);
    if (fabsf(h_force[i] - ref) > tol) {
      if (errors < 100)
        printf("*** error: body %d expected=%f, actual=%f\n", i, ref, h_force[i]);
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