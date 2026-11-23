// Parboil stencil — standalone self-checking OpenCL port for Vortex, single
// precision only.
//
// Source:
//   third_benchmark/parboil/benchmarks/stencil/src/opencl_base/kernel.cl
//   third_benchmark/parboil/benchmarks/stencil/src/opencl_base/main.c
//     (c0 = 1/6, c1 = 1/6/6, iteration loop, A0/Anext ping-pong)
//   third_benchmark/parboil/benchmarks/stencil/src/omp_base/kernels.c
//     (cpu_stencil: the host reference used here)
//
// Parboil reads a binary grid and writes a result file; this port generates a
// deterministic grid, computes the reference serial stencil and compares the
// whole n^3 volume. Both buffers are initialised equal (main.c:135), and the
// kernel writes interior cells only, so boundary cells must still hold their
// initial values — the comparison covers them and would catch out-of-range
// writes.
//
// Adaptation: upstream launches 256-work-item groups along x. Vortex caps a
// work-group at 16, so x uses 16-work-item groups with the grid rounded up;
// the kernel's `if (i < nx-1)` guard covers the extra work-items. j/k keep the
// upstream ny-2 / nz-2 grid extents.
//
// Data: deterministic rand()/RAND_MAX values in [0,1). The reference follows
// cpu_stencil's expression order; the tolerance is 1e-3 relative + 1e-3
// absolute.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "common.h"

#define KERNEL_NAME "naive_kernel"

#define LWS 16
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
cl_mem a0_memobj = NULL;
cl_mem anext_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (a0_memobj) clReleaseMemObject(a0_memobj);
  if (anext_memobj) clReleaseMemObject(anext_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 16;
int iterations = 4;

static void show_usage() {
  printf("Usage: [-n grid size] [-t iterations] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 't':
      iterations = atoi(optarg);
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

static size_t round_up(size_t value, size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

static void stencil_cpu(float c0, float c1, const float* A0, float* Anext,
                        int nx, int ny, int nz) {
  // third_benchmark/.../stencil/src/omp_base/kernels.c: cpu_stencil
  for (int i = 1; i < nx - 1; ++i) {
    for (int j = 1; j < ny - 1; ++j) {
      for (int k = 1; k < nz - 1; ++k) {
        Anext[Index3D(nx, ny, i, j, k)] =
          (A0[Index3D(nx, ny, i, j, k + 1)] +
           A0[Index3D(nx, ny, i, j, k - 1)] +
           A0[Index3D(nx, ny, i, j + 1, k)] +
           A0[Index3D(nx, ny, i, j - 1, k)] +
           A0[Index3D(nx, ny, i + 1, j, k)] +
           A0[Index3D(nx, ny, i - 1, j, k)]) * c1
          - A0[Index3D(nx, ny, i, j, k)] * c0;
      }
    }
  }
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 3) {
    printf("Error: size must be at least 3\n");
    return -1;
  }
  if (iterations < 1) {
    printf("Error: iterations must be at least 1\n");
    return -1;
  }

  printf("Grid size=%dx%dx%d, iterations=%d\n", size, size, size, iterations);

  const int nx = size, ny = size, nz = size;
  const float c0 = 1.0f / 6.0f;        // main.c:60
  const float c1 = 1.0f / 6.0f / 6.0f; // main.c:61
  const size_t grid_size = (size_t)nx * ny * nz;
  const size_t grid_nbytes = grid_size * sizeof(float);

  std::vector<float> h_a0(grid_size);
  srand(1);
  for (size_t i = 0; i < grid_size; ++i)
    h_a0[i] = static_cast<float>(rand()) / RAND_MAX;
  std::vector<float> h_anext(h_a0); // main.c:135: memcpy(Anext, A0)

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  a0_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, grid_nbytes, NULL, &_err));
  anext_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, grid_nbytes, NULL, &_err));

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

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(float), (void *)&c0));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(float), (void *)&c1));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&a0_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&anext_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void *)&nx));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void *)&ny));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), (void *)&nz));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, a0_memobj, CL_TRUE, 0, grid_nbytes, h_a0.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, anext_memobj, CL_TRUE, 0, grid_nbytes, h_anext.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t global_work_size[3] = {
    round_up((size_t)nx - 2, LWS), (size_t)ny - 2, (size_t)nz - 2
  };
  const size_t local_work_size[3] = {LWS, 1, 1};
  auto time_start = std::chrono::high_resolution_clock::now();
  size_t src = 0;
  // main.c:175-187: kernel(A0, Anext), then swap the two buffers.
  for (int t = 0; t < iterations; ++t) {
    cl_mem d_a0 = src == 0 ? a0_memobj : anext_memobj;
    cl_mem d_anext = src == 0 ? anext_memobj : a0_memobj;
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_a0));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_anext));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL,
                                    global_work_size, local_work_size,
                                    0, NULL, NULL));
    src ^= 1;
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  cl_mem d_result = src == 0 ? a0_memobj : anext_memobj;
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_result, CL_TRUE, 0, grid_nbytes,
                               h_a0.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // The device result is the buffer written by the last iteration; replay the
  // same ping-pong on the host, including the untouched boundaries.
  std::vector<float> ref_src(grid_size);
  srand(1);
  for (size_t i = 0; i < grid_size; ++i)
    ref_src[i] = static_cast<float>(rand()) / RAND_MAX;
  std::vector<float> ref_dst(ref_src);
  std::vector<float>* cur_src = &ref_src;
  std::vector<float>* cur_dst = &ref_dst;
  for (int t = 0; t < iterations; ++t) {
    stencil_cpu(c0, c1, cur_src->data(), cur_dst->data(), nx, ny, nz);
    std::swap(cur_src, cur_dst);
  }

  int errors = 0;
  for (size_t i = 0; i < grid_size; ++i) {
    float expected = (*cur_src)[i];
    float actual = h_a0[i];
    float tol = REL_TOL * fabsf(expected) + ABS_TOL;
    if (fabsf(actual - expected) > tol) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, expected, actual, fabsf(actual - expected));
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