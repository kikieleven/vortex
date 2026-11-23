// fdtd-2d (PolyBench-ACC stencils/fdtd-2d) — standalone self-checking OpenCL
// port for Vortex.
//
//   for t in 0..tmax-1:
//     ey[0][j] = fict[t]; ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]) for i>0
//     ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1])                      for j>0
//     hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j]+ey[i+1][j]-ey[i][j])
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.
// `fict[t] = t`, as in PolyBench's init_array.
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the same
// mixed tolerance form used by tests/opencl/{hotspot3D,cfd,lud}. The fields are
// differences of previous values and can be negative.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL1_NAME "fdtd_kernel1"
#define KERNEL2_NAME "fdtd_kernel2"
#define KERNEL3_NAME "fdtd_kernel3"

// Mixed absolute/relative float tolerance (see tests/opencl/hotspot3D/main.cc).
#define REL_TOL 1e-3f
#define ABS_TOL 1e-3f

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

static float generate() {
  return static_cast<float>(rand()) / RAND_MAX;
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.\n");
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

static void fdtd_cpu(float* ex, float* ey, float* hz, const float* fict,
                     int nx, int ny, int tmax) {
  for (int t = 0; t < tmax; ++t) {
    for (int j = 0; j < ny; ++j) {
      ey[0 * ny + j] = fict[t];
    }
    for (int i = 1; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        ey[i * ny + j] = ey[i * ny + j] - 0.5f * (hz[i * ny + j] - hz[(i - 1) * ny + j]);
      }
    }
    for (int i = 0; i < nx; ++i) {
      for (int j = 1; j < ny; ++j) {
        ex[i * ny + j] = ex[i * ny + j] - 0.5f * (hz[i * ny + j] - hz[i * ny + (j - 1)]);
      }
    }
    for (int i = 0; i < nx - 1; ++i) {
      for (int j = 0; j < ny - 1; ++j) {
        hz[i * ny + j] = hz[i * ny + j] - 0.7f * (ex[i * ny + (j + 1)] - ex[i * ny + j]
                                                 + ey[(i + 1) * ny + j] - ey[i * ny + j]);
      }
    }
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel1 = NULL;
cl_kernel kernel2 = NULL;
cl_kernel kernel3 = NULL;
cl_mem ex_memobj = NULL;
cl_mem ey_memobj = NULL;
cl_mem hz_memobj = NULL;
cl_mem fict_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel1) clReleaseKernel(kernel1);
  if (kernel2) clReleaseKernel(kernel2);
  if (kernel3) clReleaseKernel(kernel3);
  if (program) clReleaseProgram(program);
  if (ex_memobj) clReleaseMemObject(ex_memobj);
  if (ey_memobj) clReleaseMemObject(ey_memobj);
  if (hz_memobj) clReleaseMemObject(hz_memobj);
  if (fict_memobj) clReleaseMemObject(fict_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 16;
int tmax = 8;

static void show_usage() {
  printf("Usage: [-n size] [-t tmax] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 't':
      tmax = atoi(optarg);
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

  printf("float workload size=%d tmax=%d\n", size, tmax);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 3 || tmax < 1) {
    printf("Error: size must be at least 3 and tmax at least 1\n");
    return -1;
  }

  const int nx = size;
  const int ny = size;
  const size_t field_size = (size_t)nx * ny;
  const size_t field_nbytes = field_size * sizeof(float);
  const size_t fict_nbytes = (size_t)tmax * sizeof(float);

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  ex_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, field_nbytes, NULL, &_err));
  ey_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, field_nbytes, NULL, &_err));
  hz_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, field_nbytes, NULL, &_err));
  fict_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, fict_nbytes, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel1 = CL_CHECK2(clCreateKernel(program, KERNEL1_NAME, &_err));
  kernel2 = CL_CHECK2(clCreateKernel(program, KERNEL2_NAME, &_err));
  kernel3 = CL_CHECK2(clCreateKernel(program, KERNEL3_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&fict_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&ex_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&ey_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&hz_memobj));
  CL_CHECK(clSetKernelArg(kernel1, 5, sizeof(int), (void *)&nx));
  CL_CHECK(clSetKernelArg(kernel1, 6, sizeof(int), (void *)&ny));

  CL_CHECK(clSetKernelArg(kernel2, 0, sizeof(cl_mem), (void *)&ex_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 1, sizeof(cl_mem), (void *)&ey_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 2, sizeof(cl_mem), (void *)&hz_memobj));
  CL_CHECK(clSetKernelArg(kernel2, 3, sizeof(int), (void *)&nx));
  CL_CHECK(clSetKernelArg(kernel2, 4, sizeof(int), (void *)&ny));

  CL_CHECK(clSetKernelArg(kernel3, 0, sizeof(cl_mem), (void *)&ex_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 1, sizeof(cl_mem), (void *)&ey_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 2, sizeof(cl_mem), (void *)&hz_memobj));
  CL_CHECK(clSetKernelArg(kernel3, 3, sizeof(int), (void *)&nx));
  CL_CHECK(clSetKernelArg(kernel3, 4, sizeof(int), (void *)&ny));

  // Host-side input/output buffers
  std::vector<float> h_ex(field_size);
  std::vector<float> h_ey(field_size);
  std::vector<float> h_hz(field_size);
  std::vector<float> h_fict(tmax);
  std::vector<float> h_ex_o(field_size, 0.0f);
  std::vector<float> h_ey_o(field_size, 0.0f);
  std::vector<float> h_hz_o(field_size, 0.0f);

  // Generate input values. fict[t] = t, as in PolyBench's init_array.
  for (size_t i = 0; i < field_size; ++i) {
    h_ex[i] = generate();
    h_ey[i] = generate();
    h_hz[i] = generate();
  }
  for (int t = 0; t < tmax; ++t) {
    h_fict[t] = (float)t;
  }

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, ex_memobj, CL_TRUE, 0, field_nbytes, h_ex.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, ey_memobj, CL_TRUE, 0, field_nbytes, h_ey.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, hz_memobj, CL_TRUE, 0, field_nbytes, h_hz.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, fict_memobj, CL_TRUE, 0, fict_nbytes, h_fict.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Vortex's OpenCL device caps a work-group at 16 work-items, so use a 4x4
  // work-group (PolyBench's original 32x8 is not portable here) and round the
  // global size up to a multiple of the local size.
  size_t global_work_size[2];
  size_t local_work_size[2] = {4, 4};
  global_work_size[0] = ((ny + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
  global_work_size[1] = ((nx + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < tmax; ++t) {
    CL_CHECK(clSetKernelArg(kernel1, 4, sizeof(int), (void *)&t));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel1, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel2, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel3, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, ex_memobj, CL_TRUE, 0, field_nbytes, h_ex_o.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, ey_memobj, CL_TRUE, 0, field_nbytes, h_ey_o.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, hz_memobj, CL_TRUE, 0, field_nbytes, h_hz_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  fdtd_cpu(h_ex.data(), h_ey.data(), h_hz.data(), h_fict.data(), nx, ny, tmax);
  int errors = 0;
  for (size_t i = 0; i < field_size; ++i) {
    float a = h_ex_o[i];
    float b = h_ex[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: ex[%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  for (size_t i = 0; i < field_size; ++i) {
    float a = h_ey_o[i];
    float b = h_ey[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: ey[%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  for (size_t i = 0; i < field_size; ++i) {
    float a = h_hz_o[i];
    float b = h_hz[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: hz[%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
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
