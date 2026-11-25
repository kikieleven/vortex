// SRAD v2 (tiled) — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/gpu-rodinia/cuda/srad/srad_v2 (CUDA). Rodinia ships
// no OpenCL version of srad_v2, so this is a CUDA -> OpenCL conversion.
//
// Pipeline (same as the CUDA v2 host):
//   1. J = exp(I)                        (host; I is a deterministic [0,1) field)
//   2. per iteration:
//        srad_kernel_1 : N/S/W/E derivatives, ICOV, diffusion coefficient C
//        srad_kernel_2 : divergence of the flux, J += 0.25*lambda*D
//      the speckle statistic q0sqr is reduced on the host from the current J
//   3. compare J against a serial CPU reference running the identical
//      row-major, single-precision arithmetic.
//
// The kernels are the tiled v2 form: each 4x4 work-group stages its tile plus
// the neighbour row/column into local memory and barriers between the loads and
// the compute. rows and cols must be multiples of BLOCK_SIZE.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <CL/opencl.h>

#define CL_CHECK(_expr)                                                \
  do {                                                                 \
    cl_int _err = _expr;                                               \
    if (_err == CL_SUCCESS)                                            \
      break;                                                           \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);    \
    cleanup();                                                         \
    exit(-1);                                                          \
  } while (0)

#define CL_CHECK2(_expr)                                               \
  ({                                                                   \
    cl_int _err = CL_INVALID_VALUE;                                    \
    decltype(_expr) _ret = _expr;                                      \
    if (_err != CL_SUCCESS) {                                          \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);  \
      cleanup();                                                       \
      exit(-1);                                                        \
    }                                                                  \
    _ret;                                                              \
  })

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

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel srad_kernel_1 = NULL;
cl_kernel srad_kernel_2 = NULL;
cl_mem d_J = NULL;
cl_mem d_C = NULL, d_E = NULL, d_W = NULL, d_N = NULL, d_S = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (srad_kernel_1) clReleaseKernel(srad_kernel_1);
  if (srad_kernel_2) clReleaseKernel(srad_kernel_2);
  if (program) clReleaseProgram(program);
  if (d_J) clReleaseMemObject(d_J);
  if (d_C) clReleaseMemObject(d_C);
  if (d_E) clReleaseMemObject(d_E);
  if (d_W) clReleaseMemObject(d_W);
  if (d_N) clReleaseMemObject(d_N);
  if (d_S) clReleaseMemObject(d_S);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// BLOCK_SIZE is a compile-time constant in the kernel (local tile arrays).
static const int BLOCK_SIZE = 4;
static int Nr = 32;
static int Nc = 32;
static int niter = 2;
static float lambda = 0.5f;

static void show_usage() {
  printf("Usage: [-r rows] [-c cols] [-n niter] [-l lambda] [-h]\n"
         "  rows and cols must be multiples of %d (tiled v2 kernel).\n",
         BLOCK_SIZE);
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:n:l:h")) != -1) {
    switch (c) {
    case 'r': Nr = atoi(optarg); break;
    case 'c': Nc = atoi(optarg); break;
    case 'n': niter = atoi(optarg); break;
    case 'l': lambda = atof(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (Nr < 2 * BLOCK_SIZE || Nc < 2 * BLOCK_SIZE || niter < 1) {
    printf("Error: need rows,cols >= %d and niter >= 1\n", 2 * BLOCK_SIZE);
    exit(-1);
  }
  if ((Nr % BLOCK_SIZE) != 0 || (Nc % BLOCK_SIZE) != 0) {
    printf("Error: rows (%d) and cols (%d) must be multiples of %d\n",
           Nr, Nc, BLOCK_SIZE);
    exit(-1);
  }
}

// Serial CPU reference: exactly the CUDA v2 host's CPU path (row-major,
// stride cols, clamped N/S/W/E neighbours).
static void srad_v2_cpu(std::vector<float>& J, int rows, int cols, int niter,
                        float lambda) {
  long Ne = (long)rows * cols;
  std::vector<int> iN(rows), iS(rows), jW(cols), jE(cols);
  for (int i = 0; i < rows; ++i) { iN[i] = i - 1; iS[i] = i + 1; }
  for (int j = 0; j < cols; ++j) { jW[j] = j - 1; jE[j] = j + 1; }
  iN[0] = 0; iS[rows - 1] = rows - 1; jW[0] = 0; jE[cols - 1] = cols - 1;

  std::vector<float> dN(Ne), dS(Ne), dW(Ne), dE(Ne), cc(Ne);

  for (int iter = 0; iter < niter; ++iter) {
    float sum = 0.f, sum2 = 0.f;
    for (long k = 0; k < Ne; ++k) {
      float t = J[k];
      sum += t;
      sum2 += t * t;
    }
    float meanROI = sum / (float)Ne;
    float varROI = (sum2 / (float)Ne) - meanROI * meanROI;
    float q0sqr = varROI / (meanROI * meanROI);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int k = i * cols + j;
        float Jc = J[k];
        float dn = J[iN[i] * cols + j] - Jc;
        float ds = J[iS[i] * cols + j] - Jc;
        float dw = J[i * cols + jW[j]] - Jc;
        float de = J[i * cols + jE[j]] - Jc;

        float G2 = (dn * dn + ds * ds + dw * dw + de * de) / (Jc * Jc);
        float L = (dn + ds + dw + de) / Jc;
        float num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
        float den = 1 + (0.25f * L);
        float qsqr = num / (den * den);
        den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
        float c = 1.0f / (1.0f + den);
        if (c < 0) c = 0;
        else if (c > 1) c = 1;

        dN[k] = dn; dS[k] = ds; dW[k] = dw; dE[k] = de;
        cc[k] = c;
      }
    }

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int k = i * cols + j;
        float cN = cc[k];
        float cS = cc[iS[i] * cols + j];
        float cW = cc[k];
        float cE = cc[i * cols + jE[j]];
        float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
        J[k] = J[k] + 0.25f * lambda * D;
      }
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  long Ne = (long)Nr * Nc;
  printf("SRAD v2: Nr=%d Nc=%d Ne=%ld niter=%d lambda=%g block=%d\n",
         Nr, Nc, Ne, niter, lambda, BLOCK_SIZE);

  // Deterministic input in [0,1) via a private LCG. Do NOT use rand(): PoCL
  // shares the process-wide generator and perturbs the sequence, which would
  // make simx/rtlsim retire different instruction counts.
  std::vector<float> h_J(Ne);
  uint32_t state = 7u;
  for (long i = 0; i < Ne; ++i) {
    state = state * 1664525u + 1013904223u;
    float I = (float)(state / 4294967296.0);
    h_J[i] = expf(I);
  }

  printf("Create context\n");
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  printf("Allocate device buffers\n");
  size_t mem_size = sizeof(float) * Ne;
  d_J = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_C = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_E = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_W = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_N = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));
  d_S = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  char build_opts[64];
  snprintf(build_opts, sizeof(build_opts), "-DBLOCK_SIZE=%d", BLOCK_SIZE);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));

  srad_kernel_1 = CL_CHECK2(clCreateKernel(program, "srad_kernel_1", &_err));
  srad_kernel_2 = CL_CHECK2(clCreateKernel(program, "srad_kernel_2", &_err));
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_J, CL_TRUE, 0, mem_size,
                                h_J.data(), 0, NULL, NULL));

  size_t global_work_size[2] = {(size_t)Nc, (size_t)Nr};
  size_t local_work_size[2] = {(size_t)BLOCK_SIZE, (size_t)BLOCK_SIZE};

  // Static kernel arguments (q0sqr is refreshed every iteration).
  CL_CHECK(clSetKernelArg(srad_kernel_1, 0, sizeof(cl_mem), &d_J));
  CL_CHECK(clSetKernelArg(srad_kernel_1, 1, sizeof(cl_mem), &d_C));
  CL_CHECK(clSetKernelArg(srad_kernel_1, 2, sizeof(cl_mem), &d_E));
  CL_CHECK(clSetKernelArg(srad_kernel_1, 3, sizeof(cl_mem), &d_W));
  CL_CHECK(clSetKernelArg(srad_kernel_1, 4, sizeof(cl_mem), &d_N));
  CL_CHECK(clSetKernelArg(srad_kernel_1, 5, sizeof(cl_mem), &d_S));
  CL_CHECK(clSetKernelArg(srad_kernel_1, 6, sizeof(int), &Nc));
  CL_CHECK(clSetKernelArg(srad_kernel_1, 7, sizeof(int), &Nr));

  CL_CHECK(clSetKernelArg(srad_kernel_2, 0, sizeof(cl_mem), &d_J));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 1, sizeof(cl_mem), &d_C));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 2, sizeof(cl_mem), &d_E));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 3, sizeof(cl_mem), &d_W));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 4, sizeof(cl_mem), &d_N));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 5, sizeof(cl_mem), &d_S));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 6, sizeof(int), &Nc));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 7, sizeof(int), &Nr));
  CL_CHECK(clSetKernelArg(srad_kernel_2, 8, sizeof(float), &lambda));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();

  std::vector<float> host_J(Ne);
  for (int iter = 0; iter < niter; ++iter) {
    // Statistics reduction on the host (same as the CUDA v2 host).
    CL_CHECK(clEnqueueReadBuffer(commandQueue, d_J, CL_TRUE, 0, mem_size,
                                 host_J.data(), 0, NULL, NULL));
    float sum = 0.f, sum2 = 0.f;
    for (long k = 0; k < Ne; ++k) {
      float t = host_J[k];
      sum += t;
      sum2 += t * t;
    }
    float meanROI = sum / (float)Ne;
    float varROI = (sum2 / (float)Ne) - meanROI * meanROI;
    float q0sqr = varROI / (meanROI * meanROI);

    CL_CHECK(clSetKernelArg(srad_kernel_1, 8, sizeof(float), &q0sqr));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, srad_kernel_1, 2, NULL,
                                    global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, srad_kernel_2, 2, NULL,
                                    global_work_size, local_work_size, 0, NULL, NULL));
  }

  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  std::vector<float> h_gpu(Ne);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_J, CL_TRUE, 0, mem_size,
                               h_gpu.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<float> h_ref(h_J);
  srad_v2_cpu(h_ref, Nr, Nc, niter, lambda);

  const float atol = 1e-2f;
  const float rtol = 1e-3f;
  int errors = 0;
  for (long i = 0; i < Ne; ++i) {
    float a = h_ref[i], b = h_gpu[i];
    if (fabsf(a - b) > atol + rtol * fabsf(a)) {
      if (errors < 100)
        printf("*** error: [%ld] expected=%f, actual=%f\n", i, a, b);
      ++errors;
    }
  }

  cleanup();
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}