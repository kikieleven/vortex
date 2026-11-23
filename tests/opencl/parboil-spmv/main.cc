// Parboil spmv — standalone self-checking OpenCL port for Vortex, single
// precision only.
//
// Source:
//   third_benchmark/parboil/benchmarks/spmv/src/opencl_base/kernel.cl
//   third_benchmark/parboil/benchmarks/spmv/src/opencl_base/main.c
//   third_benchmark/parboil/benchmarks/spmv/common_src/convert-dataset/convert_dataset.c
//     (coo_to_jds: padded JDS conversion; warp_size=32, pack_size=1, pad_rows=1)
//
// Parboil reads a Matrix Market file and writes the result vector; this port
// generates the JDS matrix directly and adds a host reference.
//
// JDS construction used here (convert_dataset.c, with pad_rows=1/pack_size=1
// so no padding entries are needed):
//   - the problem dimension is rounded up to a multiple of the 32-row warp
//     group used by the kernel;
//   - every row has the same number of non-zeros (DEGREE), so the per-group
//     bound sh_zcnt_int[g] is DEGREE for every group and no row needs the
//     zero/-1 padding entries the converter emits;
//   - data/index are diagonal-major: diagonal k occupies [k*dim, (k+1)*dim);
//   - perm is the JDS-row -> original-row map and is non-trivial, so the
//     output indirection is exercised.
//
// Adaptation: upstream picks the grid/block from device properties; Vortex
// caps a work-group at 16, so the kernel is launched 1D with 16-work-item
// groups.
//
// Data: deterministic rand()/RAND_MAX values in [0,1); the reference sums the
// same non-zeros in the same ascending-k order, so the tolerance is 1e-5
// relative + 1e-6 absolute.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "spmv_jds_naive"

#define LWS 16
#define WARP_GROUP 32
#define DEGREE 4
#define REL_TOL 1e-5f
#define ABS_TOL 1e-6f

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
cl_mem data_memobj = NULL;
cl_mem index_memobj = NULL;
cl_mem perm_memobj = NULL;
cl_mem x_memobj = NULL;
cl_mem dst_memobj = NULL;
cl_mem ptr_memobj = NULL;
cl_mem zcnt_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (data_memobj) clReleaseMemObject(data_memobj);
  if (index_memobj) clReleaseMemObject(index_memobj);
  if (perm_memobj) clReleaseMemObject(perm_memobj);
  if (x_memobj) clReleaseMemObject(x_memobj);
  if (dst_memobj) clReleaseMemObject(dst_memobj);
  if (ptr_memobj) clReleaseMemObject(ptr_memobj);
  if (zcnt_memobj) clReleaseMemObject(zcnt_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 256;

static void show_usage() {
  printf("Usage: [-n matrix rows] [-h: help]\n");
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

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 1) {
    printf("Error: size must be positive\n");
    return -1;
  }

  // convert_dataset.c pads the row count up to the warp group size.
  const int dim = (size + WARP_GROUP - 1) / WARP_GROUP * WARP_GROUP;
  const int num_groups = dim / WARP_GROUP;

  printf("Rows=%d (padded to %d), non-zeros per row=%d\n", size, dim, DEGREE);

  // perm is a bijection; with gcd(3, dim) == 1 (dim is a multiple of 32) the
  // affine map below is a permutation.
  std::vector<int> h_perm(dim);
  for (int i = 0; i < dim; ++i)
    h_perm[i] = (i * 3 + 1) % dim;

  std::vector<float> h_data((size_t)DEGREE * dim);
  std::vector<int> h_index((size_t)DEGREE * dim);
  srand(54321);
  for (int k = 0; k < DEGREE; ++k) {
    for (int i = 0; i < dim; ++i) {
      h_data[(size_t)k * dim + i] = static_cast<float>(rand()) / RAND_MAX;
      // distinct columns within a row because 7*k < dim for dim >= 32
      h_index[(size_t)k * dim + i] = (i + 7 * k + 3) % dim;
    }
  }

  std::vector<int> h_ptr(DEGREE);
  std::vector<int> h_zcnt(num_groups, DEGREE);
  for (int k = 0; k < DEGREE; ++k)
    h_ptr[k] = k * dim;

  std::vector<float> h_x(dim);
  for (int i = 0; i < dim; ++i)
    h_x[i] = static_cast<float>(rand()) / RAND_MAX;

  std::vector<float> h_y(dim, 0.0f);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  data_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, h_data.size() * sizeof(float), NULL, &_err));
  index_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, h_index.size() * sizeof(int), NULL, &_err));
  perm_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(int), NULL, &_err));
  x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, dim * sizeof(float), NULL, &_err));
  dst_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, dim * sizeof(float), NULL, &_err));
  ptr_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, DEGREE * sizeof(int), NULL, &_err));
  zcnt_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, num_groups * sizeof(int), NULL, &_err));

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

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&dst_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&index_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&perm_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&x_memobj));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void *)&dim));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&ptr_memobj));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&zcnt_memobj));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, data_memobj, CL_TRUE, 0, h_data.size() * sizeof(float), h_data.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, index_memobj, CL_TRUE, 0, h_index.size() * sizeof(int), h_index.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, perm_memobj, CL_TRUE, 0, dim * sizeof(int), h_perm.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, dim * sizeof(float), h_x.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, dst_memobj, CL_TRUE, 0, dim * sizeof(float), h_y.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, ptr_memobj, CL_TRUE, 0, DEGREE * sizeof(int), h_ptr.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, zcnt_memobj, CL_TRUE, 0, num_groups * sizeof(int), h_zcnt.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t global_work_size = (size_t)dim;
  const size_t local_work_size = LWS;
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, dst_memobj, CL_TRUE, 0, dim * sizeof(float),
                               h_y.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<float> h_ref(dim);
  for (int i = 0; i < dim; ++i) {
    float sum = 0.0f;
    for (int k = 0; k < DEGREE; ++k) {
      size_t j = (size_t)k * dim + i;
      sum += h_data[j] * h_x[h_index[j]];
    }
    h_ref[h_perm[i]] = sum;
  }

  int errors = 0;
  for (int i = 0; i < dim; ++i) {
    float expected = h_ref[i];
    float actual = h_y[i];
    float tol = REL_TOL * fabsf(expected) + ABS_TOL;
    if (fabsf(actual - expected) > tol) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f (diff=%g)\n",
               i, expected, actual, fabsf(actual - expected));
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