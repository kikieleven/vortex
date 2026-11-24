// SHOC Stencil2D (level1/stencil2d) — standalone self-checking OpenCL port for
// Vortex, single precision only.
//
// Source:
//   third_benchmark/shoc/src/opencl/level1/stencil2d/stencil2d.cl
//   third_benchmark/shoc/src/opencl/level1/stencil2d/OpenCLStencil.cpp:230-420
//     (buffer layout, halo handling, launch dimensions)
//   third_benchmark/shoc/src/common/HostStencil.cpp (host reference)
//   third_benchmark/shoc/src/common/InitializeMatrix2D.cpp (initial data)
//   third_benchmark/shoc/src/common/Matrix2D.h (row pitch: round_up(cols, 16))
//
// The kernel file is the SHOC original. Its precision and tile macros are not
// source-level defines here: LROWS/LCOLS and SINGLE_PRECISION are passed through
// clBuildProgram options, which PoCL's Stage 1 does see (see the port guide
// §4.5). SHOC's default lsize is 8x256; Vortex caps a work-group at 16, so this
// port uses LROWS=4 strips with LCOLS=16 columns (local size {1,16}).
//
// Matrix layout is SHOC's: a haloed (n+2)x(n+2) logical matrix stored row-major
// with a row pitch of round_up(n+2, 16). Only the interior is computed; the halo
// stays at haloVal (0.0).
//
// Host reference: the same 9-point stencil with the same association order as
// the kernel (cardinal N+S+E+W, diagonal NE+SE+NW+SW). SHOC's HostStencil uses
// N+S+W+E and NW+SW+NE+SE; both are algebraically identical but not
// bit-identical in float, and SHOC validates with a 0.01 threshold. This port
// uses the kernel's order and the mixed abs/rel tolerance so the comparison can
// be tighter.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "StencilKernel"

// Vortex's OpenCL device caps a work-group at 16 work-items.
#define LROWS 4
#define LCOLS 16
#define MATRIX_PAD 16

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

static size_t round_up(size_t v, size_t a) {
  return v + ((v % a == 0) ? 0 : (a - (v % a)));
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem data_a_memobj = NULL;
cl_mem data_b_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (data_a_memobj) clReleaseMemObject(data_a_memobj);
  if (data_b_memobj) clReleaseMemObject(data_b_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int grid_size = 64;
int n_iters = 2;

static void show_usage() {
  printf("Usage: [-n interior size] [-t iterations] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:h")) != -1) {
    switch (c) {
    case 'n':
      grid_size = atoi(optarg);
      break;
    case 't':
      n_iters = atoi(optarg);
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

  printf("float workload size=%d iterations=%d\n", grid_size, n_iters);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // The kernel processes LROWS rows per group and LCOLS columns per group, and
  // its index space is the logical (non-halo) matrix.
  if (grid_size < 2 || (grid_size % LROWS) != 0 || (grid_size % LCOLS) != 0) {
    printf("Error: size must be a multiple of %d and %d, and at least 2\n",
           LROWS, LCOLS);
    return -1;
  }
  if (n_iters < 1) {
    printf("Error: iterations must be at least 1\n");
    return -1;
  }

  const int n = grid_size;
  const int rows = n + 2;   // includes a 1-wide halo
  const int cols = n + 2;
  const int pitch = (int)round_up((size_t)cols, MATRIX_PAD);
  const size_t nbytes = (size_t)rows * pitch * sizeof(float);

  const float w_center = 0.25f;    // SHOC defaults (Stencil2Dmain.cpp:409-411)
  const float w_cardinal = 0.15f;
  const float w_diagonal = 0.05f;
  const float halo_val = 0.0f;

  // InitializeMatrix2D with READY undefined: interior = i*j, halo = haloVal.
  std::vector<float> h_init((size_t)rows * pitch, 0.0f);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      h_init[(size_t)(i + 1) * pitch + (j + 1)] = (float)(i * j);
  // halo ring stays halo_val (0.0f); padding columns are also 0 here.

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

  // The kernel's precision/tile macros must reach PoCL Stage 1, so they are
  // build options rather than host -D flags (port guide §4.5).
  char build_opts[128];
  snprintf(build_opts, sizeof(build_opts),
           "-DSINGLE_PRECISION -DLROWS=%d -DLCOLS=%d", LROWS, LCOLS);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  data_a_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));
  data_b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, nbytes, NULL, &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, data_a_memobj, CL_TRUE, 0, nbytes,
                                h_init.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, data_b_memobj, CL_TRUE, 0, nbytes,
                                h_init.data(), 0, NULL, NULL));

  const int alignment = MATRIX_PAD;
  const size_t local_data_size = (size_t)(LROWS + 2) * (LCOLS + 2) * sizeof(float);

  const size_t global_work_size[2] = {(size_t)(n / LROWS), (size_t)n};
  const size_t local_work_size[2] = {1, LCOLS};

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();

  cl_mem result_memobj = data_a_memobj;
  for (int iter = 0; iter < n_iters; ++iter) {
    cl_mem curr = (iter % 2 == 0) ? data_a_memobj : data_b_memobj;
    cl_mem next = (iter % 2 == 0) ? data_b_memobj : data_a_memobj;

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&curr));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&next));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void *)&alignment));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(float), (void *)&w_center));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float), (void *)&w_cardinal));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float), (void *)&w_diagonal));
    CL_CHECK(clSetKernelArg(kernel, 6, local_data_size, NULL));

    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                                    global_work_size, local_work_size,
                                    0, NULL, NULL));
    result_memobj = next;
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  std::vector<float> h_out((size_t)rows * pitch);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, result_memobj, CL_TRUE, 0, nbytes,
                               h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Host reference: same iteration count and the kernel's association order.
  std::vector<float> h_ref(h_init);
  std::vector<float> h_tmp((size_t)rows * pitch);
  for (int iter = 0; iter < n_iters; ++iter) {
    h_tmp = h_ref;
    for (int i = 1; i < rows - 1; ++i) {
      for (int j = 1; j < cols - 1; ++j) {
        float center = h_tmp[(size_t)i * pitch + j];
        float cardinal = h_tmp[(size_t)(i - 1) * pitch + j] +
                         h_tmp[(size_t)(i + 1) * pitch + j] +
                         h_tmp[(size_t)i * pitch + (j + 1)] +
                         h_tmp[(size_t)i * pitch + (j - 1)];
        float diagonal = h_tmp[(size_t)(i - 1) * pitch + (j + 1)] +
                         h_tmp[(size_t)(i + 1) * pitch + (j + 1)] +
                         h_tmp[(size_t)(i - 1) * pitch + (j - 1)] +
                         h_tmp[(size_t)(i + 1) * pitch + (j - 1)];
        h_ref[(size_t)i * pitch + j] =
            w_center * center + w_cardinal * cardinal + w_diagonal * diagonal;
      }
    }
  }

  int errors = 0;
  for (int i = 1; i < rows - 1; ++i) {
    for (int j = 1; j < cols - 1; ++j) {
      float expected = h_ref[(size_t)i * pitch + j];
      float actual = h_out[(size_t)i * pitch + j];
      float tol = REL_TOL * fabsf(expected) + ABS_TOL;
      if (fabsf(actual - expected) > tol) {
        if (errors < 100) {
          printf("*** error: [%d,%d] expected=%f, actual=%f (diff=%g)\n",
                 i, j, expected, actual, fabsf(actual - expected));
        }
        ++errors;
      }
    }
  }

  // The halo must be untouched by the kernel.
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      bool in_halo = (i == 0) || (i == rows - 1) || (j == 0) || (j == cols - 1);
      if (!in_halo) continue;
      float actual = h_out[(size_t)i * pitch + j];
      if (actual != halo_val) {
        if (errors < 100)
          printf("*** error: halo [%d,%d] expected=%f, actual=%f\n",
                 i, j, halo_val, actual);
        ++errors;
      }
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
