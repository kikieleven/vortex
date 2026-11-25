// Parboil mri-q — standalone self-checking OpenCL port for Vortex, single
// precision only.
//
// Source:
//   third_benchmark/parboil/benchmarks/mri-q/src/opencl/kernels.cl
//   third_benchmark/parboil/benchmarks/mri-q/src/opencl/main.c
//   third_benchmark/parboil/benchmarks/mri-q/src/opencl/computeQ.c
//     (PhiMag kernel, then ComputeQ tiled over KERNEL_Q_K_ELEMS_PER_GRID)
//   third_benchmark/parboil/benchmarks/mri-q/src/cpu/computeQ.cc
//     (ComputePhiMagCPU / ComputeQCPU: the host reference used here)
//
// Parboil reads k-space data and writes Q to a file; this port generates
// deterministic data and compares Qr/Qi against the serial CPU reference.
//
// Adaptation: upstream uses 256-work-item groups for both kernels. Vortex
// caps a work-group at 16, so common.h sets ComputePhiMag's block to 16 and
// ComputeQ's block to 64 (NC=4 elements per work-item => 16 work-items). The
// host pads numX up to a multiple of 64, as upstream's launch geometry
// already assumes.
//
// Data: deterministic values in [-1,1). Both sides accumulate over k in
// ascending order; the tolerance is 1e-3 relative + 1e-3 absolute.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "common.h"

#define KERNEL_PHI_MAG "ComputePhiMag_GPU"
#define KERNEL_Q "ComputeQ_GPU"

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
cl_kernel phi_mag_kernel = NULL;
cl_kernel q_kernel = NULL;
cl_mem phiR_memobj = NULL;
cl_mem phiI_memobj = NULL;
cl_mem phiMag_memobj = NULL;
cl_mem x_memobj = NULL;
cl_mem y_memobj = NULL;
cl_mem z_memobj = NULL;
cl_mem qr_memobj = NULL;
cl_mem qi_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (phi_mag_kernel) clReleaseKernel(phi_mag_kernel);
  if (q_kernel) clReleaseKernel(q_kernel);
  if (program) clReleaseProgram(program);
  if (phiR_memobj) clReleaseMemObject(phiR_memobj);
  if (phiI_memobj) clReleaseMemObject(phiI_memobj);
  if (phiMag_memobj) clReleaseMemObject(phiMag_memobj);
  if (x_memobj) clReleaseMemObject(x_memobj);
  if (y_memobj) clReleaseMemObject(y_memobj);
  if (z_memobj) clReleaseMemObject(z_memobj);
  if (qr_memobj) clReleaseMemObject(qr_memobj);
  if (qi_memobj) clReleaseMemObject(qi_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 64;

static void show_usage() {
  printf("Usage: [-n number of X/K samples] [-h: help]\n");
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

  // Upstream sizes the Q launch as a whole number of blocks
  // (computeQ.c:60-68).
  const int numK = (size + KERNEL_Q_THREADS_PER_BLOCK - 1)
                 / KERNEL_Q_THREADS_PER_BLOCK * KERNEL_Q_THREADS_PER_BLOCK;
  const int numX = numK;

  printf("Samples: numK=%d numX=%d\n", numK, numX);

  std::vector<float> h_kx(numK), h_ky(numK), h_kz(numK);
  std::vector<float> h_x(numX), h_y(numX), h_z(numX);
  std::vector<float> h_phiR(numK), h_phiI(numK);
  srand(1);
  for (int i = 0; i < numK; ++i) {
    h_kx[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    h_ky[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    h_kz[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    h_phiR[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    h_phiI[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  }
  for (int i = 0; i < numX; ++i) {
    h_x[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    h_y[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
    h_z[i] = 2.0f * static_cast<float>(rand()) / RAND_MAX - 1.0f;
  }

  std::vector<float> h_phiMag(numK, 0.0f);
  std::vector<float> h_qr(numX, 0.0f), h_qi(numX, 0.0f);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  phiR_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, numK * sizeof(float), NULL, &_err));
  phiI_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, numK * sizeof(float), NULL, &_err));
  phiMag_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, numK * sizeof(float), NULL, &_err));
  x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, numX * sizeof(float), NULL, &_err));
  y_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, numX * sizeof(float), NULL, &_err));
  z_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, numX * sizeof(float), NULL, &_err));
  qr_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, numX * sizeof(float), NULL, &_err));
  qi_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, numX * sizeof(float), NULL, &_err));

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

  phi_mag_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_PHI_MAG, &_err));
  q_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_Q, &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, phiR_memobj, CL_TRUE, 0, numK * sizeof(float), h_phiR.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, phiI_memobj, CL_TRUE, 0, numK * sizeof(float), h_phiI.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, numX * sizeof(float), h_x.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, y_memobj, CL_TRUE, 0, numX * sizeof(float), h_y.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, z_memobj, CL_TRUE, 0, numX * sizeof(float), h_z.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, qr_memobj, CL_TRUE, 0, numX * sizeof(float), h_qr.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, qi_memobj, CL_TRUE, 0, numX * sizeof(float), h_qi.data(), 0, NULL, NULL));

  auto time_start = std::chrono::high_resolution_clock::now();

  printf("Execute PhiMag kernel\n");
  {
    size_t global = (size_t)((numK + KERNEL_PHI_MAG_THREADS_PER_BLOCK - 1)
                    / KERNEL_PHI_MAG_THREADS_PER_BLOCK * KERNEL_PHI_MAG_THREADS_PER_BLOCK);
    size_t local = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
    CL_CHECK(clSetKernelArg(phi_mag_kernel, 0, sizeof(cl_mem), (void *)&phiR_memobj));
    CL_CHECK(clSetKernelArg(phi_mag_kernel, 1, sizeof(cl_mem), (void *)&phiI_memobj));
    CL_CHECK(clSetKernelArg(phi_mag_kernel, 2, sizeof(cl_mem), (void *)&phiMag_memobj));
    CL_CHECK(clSetKernelArg(phi_mag_kernel, 3, sizeof(int), (void *)&numK));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, phi_mag_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
  }

  printf("Download phiMag\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, phiMag_memobj, CL_TRUE, 0, numK * sizeof(float),
                               h_phiMag.data(), 0, NULL, NULL));

  // main.c:190-198: build the kValues array from the device-computed phiMag.
  std::vector<struct kValues> kVals(numK);
  for (int k = 0; k < numK; ++k) {
    kVals[k].Kx = h_kx[k];
    kVals[k].Ky = h_ky[k];
    kVals[k].Kz = h_kz[k];
    kVals[k].PhiMag = h_phiMag[k];
  }

  printf("Execute Q kernel\n");
  {
    const int QGrids = (numK + KERNEL_Q_K_ELEMS_PER_GRID - 1) / KERNEL_Q_K_ELEMS_PER_GRID;
    const int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
    const size_t dimQBlock = KERNEL_Q_THREADS_PER_BLOCK / NC;
    const size_t dimQGrid = (size_t)QBlocks * KERNEL_Q_THREADS_PER_BLOCK / NC;

    cl_mem ck_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
      KERNEL_Q_K_ELEMS_PER_GRID * sizeof(struct kValues), NULL, &_err));

    for (int qgrid = 0; qgrid < QGrids; ++qgrid) {
      int qGridBase = qgrid * KERNEL_Q_K_ELEMS_PER_GRID;
      struct kValues* kValsTile = kVals.data() + qGridBase;
      int numElems = KERNEL_Q_K_ELEMS_PER_GRID;
      if (numK - qGridBase < numElems)
        numElems = numK - qGridBase;

      CL_CHECK(clEnqueueWriteBuffer(commandQueue, ck_memobj, CL_TRUE, 0,
                                    numElems * sizeof(struct kValues), kValsTile, 0, NULL, NULL));

      CL_CHECK(clSetKernelArg(q_kernel, 0, sizeof(int), (void *)&numK));
      CL_CHECK(clSetKernelArg(q_kernel, 1, sizeof(int), (void *)&qGridBase));
      CL_CHECK(clSetKernelArg(q_kernel, 2, sizeof(cl_mem), (void *)&x_memobj));
      CL_CHECK(clSetKernelArg(q_kernel, 3, sizeof(cl_mem), (void *)&y_memobj));
      CL_CHECK(clSetKernelArg(q_kernel, 4, sizeof(cl_mem), (void *)&z_memobj));
      CL_CHECK(clSetKernelArg(q_kernel, 5, sizeof(cl_mem), (void *)&qr_memobj));
      CL_CHECK(clSetKernelArg(q_kernel, 6, sizeof(cl_mem), (void *)&qi_memobj));
      CL_CHECK(clSetKernelArg(q_kernel, 7, sizeof(cl_mem), (void *)&ck_memobj));

      CL_CHECK(clEnqueueNDRangeKernel(commandQueue, q_kernel, 1, NULL,
                                      &dimQGrid, &dimQBlock, 0, NULL, NULL));
    }
    CL_CHECK(clFinish(commandQueue));
    clReleaseMemObject(ck_memobj);
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffers\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, qr_memobj, CL_TRUE, 0, numX * sizeof(float),
                               h_qr.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, qi_memobj, CL_TRUE, 0, numX * sizeof(float),
                               h_qi.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // cpu/computeQ.cc: ComputePhiMagCPU + ComputeQCPU.
  std::vector<float> ref_phiMag(numK);
  for (int k = 0; k < numK; ++k)
    ref_phiMag[k] = h_phiR[k] * h_phiR[k] + h_phiI[k] * h_phiI[k];

  std::vector<float> ref_qr(numX, 0.0f), ref_qi(numX, 0.0f);
  for (int k = 0; k < numK; ++k) {
    for (int i = 0; i < numX; ++i) {
      float expArg = PIx2 * (kVals[k].Kx * h_x[i] +
                             kVals[k].Ky * h_y[i] +
                             kVals[k].Kz * h_z[i]);
      float cosArg = cosf(expArg);
      float sinArg = sinf(expArg);
      float phi = ref_phiMag[k];
      ref_qr[i] += phi * cosArg;
      ref_qi[i] += phi * sinArg;
    }
  }

  int errors = 0;
  for (int i = 0; i < numX; ++i) {
    float expected = ref_qr[i];
    float actual = h_qr[i];
    float tol = REL_TOL * fabsf(expected) + ABS_TOL;
    if (fabsf(actual - expected) > tol) {
      if (errors < 100)
        printf("*** error: Qr[%d] expected=%f, actual=%f (diff=%g)\n",
               i, expected, actual, fabsf(actual - expected));
      ++errors;
    }
    expected = ref_qi[i];
    actual = h_qi[i];
    tol = REL_TOL * fabsf(expected) + ABS_TOL;
    if (fabsf(actual - expected) > tol) {
      if (errors < 100)
        printf("*** error: Qi[%d] expected=%f, actual=%f (diff=%g)\n",
               i, expected, actual, fabsf(actual - expected));
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