// SHOC SPMV (level1/spmv) — standalone self-checking OpenCL port for Vortex,
// single precision only, buffer path (no OpenCL images).
//
// Source:
//   third_benchmark/shoc/src/opencl/level1/spmv/spmv.cl
//     (spmv_csr_scalar_kernel)
//   third_benchmark/shoc/src/opencl/level1/spmv/Spmv.cpp:47-90   (spmvCpu)
//   third_benchmark/shoc/src/common/Spmv/util.h:244-251 (fill),
//     :272-305 (initRandomMatrix), :15 (MAX_RELATIVE_ERROR = 0.02)
//
// SHOC runs three kernels: CSR scalar, CSR vector and ELLPACKR. This port keeps
// the CSR scalar kernel (one work-item per row). The vector kernel needs a
// warp-sized group and a hard-coded 128-lane partial-sum array; ELLPACKR needs
// padded/column-major data. Both would have to be rewritten for Vortex's
// 16-work-item groups, which is out of scope for this port.
//
// SHOC's image path (USE_TEXTURE) is not ported; this is the global-memory
// scalar path.
//
// Random data: SHOC fills values with rand()*maxval and positions with
// drand48() seeded at 8675309, then fills the dense vector with rand() again
// (Spmv.cpp:836,870). The same order is reproduced here.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <algorithm>

#define KERNEL_NAME "spmv_csr_scalar_kernel"

// Vortex's OpenCL device caps a work-group at 16 work-items.
#define LWS 16

// SHOC util.h:15 and Spmv.cpp:39.
#define MAX_RELATIVE_ERROR 0.02
#define MAXVAL 10.0f

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

// SHOC util.h:244 fill(): rand() in [0, maxval).
template <typename T>
static void fill(T* a, int n, float maxi) {
  for (int j = 0; j < n; ++j)
    a[j] = (T)(maxi * (rand() / (RAND_MAX + 1.0f)));
}

// SHOC util.h:272 initRandomMatrix().
static void init_random_matrix(std::vector<int>& cols,
                               std::vector<int>& row_delimiters,
                               int n, int dim) {
  int nnz_assigned = 0;
  double prob = (double)n / ((double)dim * (double)dim);

  srand48(8675309L);

  bool fill_remaining = false;
  for (int i = 0; i < dim; ++i) {
    row_delimiters[i] = nnz_assigned;
    for (int j = 0; j < dim; ++j) {
      int num_entries_left = (dim * dim) - ((i * dim) + j);
      int need_to_assign = n - nnz_assigned;
      if (num_entries_left <= need_to_assign)
        fill_remaining = true;
      if ((nnz_assigned < n && drand48() <= prob) || fill_remaining) {
        cols[nnz_assigned] = j;
        ++nnz_assigned;
      }
    }
  }
  row_delimiters[dim] = n;
  assert(nnz_assigned == n);
}

// SHOC Spmv.cpp:47 spmvCpu.
static void spmv_cpu(const std::vector<float>& val,
                     const std::vector<int>& cols,
                     const std::vector<int>& row_delimiters,
                     const std::vector<float>& vec,
                     int dim, std::vector<float>& out) {
  for (int i = 0; i < dim; ++i) {
    float t = 0;
    for (int j = row_delimiters[i]; j < row_delimiters[i + 1]; ++j)
      t += val[j] * vec[cols[j]];
    out[i] = t;
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem val_memobj = NULL;
cl_mem vec_memobj = NULL;
cl_mem cols_memobj = NULL;
cl_mem rowdelim_memobj = NULL;
cl_mem out_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (val_memobj) clReleaseMemObject(val_memobj);
  if (vec_memobj) clReleaseMemObject(vec_memobj);
  if (cols_memobj) clReleaseMemObject(cols_memobj);
  if (rowdelim_memobj) clReleaseMemObject(rowdelim_memobj);
  if (out_memobj) clReleaseMemObject(out_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int nrows = 256;

static void show_usage() {
  printf("Usage: [-n rows] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      nrows = atoi(optarg);
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

  printf("float workload rows=%d\n", nrows);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (nrows < 1) {
    printf("Error: rows must be at least 1\n");
    return -1;
  }

  const int dim = nrows;
  const int n_items = dim * dim / 100;  // SHOC: 1% of entries non-zero
  if (n_items < 1) {
    printf("Error: matrix is too small to have any non-zero entries\n");
    return -1;
  }

  std::vector<float> h_val(n_items);
  std::vector<int> h_cols(n_items);
  std::vector<int> h_rowdelim(dim + 1);
  std::vector<float> h_vec(dim);
  std::vector<float> h_out(dim);
  std::vector<float> h_ref(dim);

  // SHOC Spmv.cpp:833-836,870: fill values, place them, then fill the vector.
  fill(h_val.data(), n_items, MAXVAL);
  init_random_matrix(h_cols, h_rowdelim, n_items, dim);
  fill(h_vec.data(), dim, MAXVAL);

  spmv_cpu(h_val, h_cols, h_rowdelim, h_vec, dim, h_ref);

  const size_t val_nbytes = (size_t)n_items * sizeof(float);
  const size_t cols_nbytes = (size_t)n_items * sizeof(int);
  const size_t rowdelim_nbytes = (size_t)(dim + 1) * sizeof(int);
  const size_t vec_nbytes = (size_t)dim * sizeof(float);
  const size_t out_nbytes = (size_t)dim * sizeof(float);

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

  val_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, val_nbytes, NULL, &_err));
  vec_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, vec_nbytes, NULL, &_err));
  cols_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, cols_nbytes, NULL, &_err));
  rowdelim_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, rowdelim_nbytes, NULL, &_err));
  out_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, out_nbytes, NULL, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&val_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&vec_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&cols_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&rowdelim_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void *)&dim));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&out_memobj));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, val_memobj, CL_TRUE, 0, val_nbytes,
                                h_val.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, vec_memobj, CL_TRUE, 0, vec_nbytes,
                                h_vec.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, cols_memobj, CL_TRUE, 0, cols_nbytes,
                                h_cols.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, rowdelim_memobj, CL_TRUE, 0, rowdelim_nbytes,
                                h_rowdelim.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t local_work_size = LWS;
  const size_t global_work_size = ((size_t)dim + LWS - 1) / LWS * LWS;
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, out_memobj, CL_TRUE, 0, out_nbytes,
                               h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  for (int i = 0; i < dim; ++i) {
    // SHOC verifyResults divides by the CPU result; rows with no non-zeros
    // would divide by zero, so an absolute term is added.
    float cpu = h_ref[i];
    float gpu = h_out[i];
    float tol = (float)MAX_RELATIVE_ERROR * fabsf(cpu) + 1e-6f;
    if (fabsf(cpu - gpu) > tol) {
      if (errors < 100) {
        printf("*** error: row %d expected=%f, actual=%f\n", i, cpu, gpu);
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
