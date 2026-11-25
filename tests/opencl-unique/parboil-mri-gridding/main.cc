// Parboil mri-gridding — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/parboil/benchmarks/mri-gridding/src/opencl_base/
//     GPU_kernels.cl      (binning, reorder, gridding)
//     scanLargeArray.cl   (block/inter/uniform-add scan)
//     OpenCL_interface.cpp (pipeline order)
//     CPU_kernels.c       (gridding_Gold: host reference)
//     main.cpp            (parameters and sample layout)
//
// Pipeline per run: bin on the GPU, sort the (key,value) pairs, reorder the
// samples, scan the bin counts into start addresses, grid on the GPU, and let
// the host add the contributions of samples that overflowed a bin.
//
// Deviation: the radix-sort stage (sort.cl) runs on the host as a stable sort
// of the same (key,value) pairs. sort.cl is built around uint4 loads and a
// 4-elements-per-thread block scan; re-blocking it for 16 work-items would
// replace its parallel structure, and the vector loads cannot be selected by
// the RISC-V backend. A stable sort by key produces the same ordered arrays
// that the pipeline consumes, so the grid result is unchanged; the sort kernels
// themselves are NOT covered by this test (recorded as unverified).
//
// Adaptations (see kernel.cl): 4x2x2 = 16-work-item gridding groups with a
// looped TILE load, flat float grid data, runtime grid constants as arguments,
// 16-work-item scan blocks. Atomics need the Vortex A extension.
//
// Data: grid 8x8x8, 64 samples on a 4x4x4 lattice (at most one per bin, so the
// overflow/CPU path is empty), kernelWidth 4, oversample 2. The reference is
// the full CPU gridding sum; comparison is 1e-3 relative + 1e-3 absolute.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <algorithm>

#include "common.h"

#define KERNEL_BINNING "binning_kernel"
#define KERNEL_REORDER "reorder_kernel"
#define KERNEL_GRID "gridding_GPU"
#define KERNEL_SCAN_L1 "scan_L1_kernel"
#define KERNEL_SCAN_I1 "scan_inter1_kernel"
#define KERNEL_SCAN_I2 "scan_inter2_kernel"
#define KERNEL_UNIFORM_ADD "uniformAdd"

#define GROUP_SIZE 16
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
  fseek(fp, 0, SEEK_SET);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel binning_kernel = NULL;
cl_kernel reorder_kernel = NULL;
cl_kernel grid_kernel = NULL;
cl_kernel scan_l1_kernel = NULL;
cl_kernel scan_i1_kernel = NULL;
cl_kernel scan_i2_kernel = NULL;
cl_kernel uniform_add_kernel = NULL;
cl_mem sample_memobj = NULL;
cl_mem key_memobj = NULL;
cl_mem value_memobj = NULL;
cl_mem bin_start_memobj = NULL;
cl_mem sorted_memobj = NULL;
cl_mem griddata_memobj = NULL;
cl_mem density_memobj = NULL;
cl_mem inter_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (binning_kernel) clReleaseKernel(binning_kernel);
  if (reorder_kernel) clReleaseKernel(reorder_kernel);
  if (grid_kernel) clReleaseKernel(grid_kernel);
  if (scan_l1_kernel) clReleaseKernel(scan_l1_kernel);
  if (scan_i1_kernel) clReleaseKernel(scan_i1_kernel);
  if (scan_i2_kernel) clReleaseKernel(scan_i2_kernel);
  if (uniform_add_kernel) clReleaseKernel(uniform_add_kernel);
  if (program) clReleaseProgram(program);
  if (sample_memobj) clReleaseMemObject(sample_memobj);
  if (key_memobj) clReleaseMemObject(key_memobj);
  if (value_memobj) clReleaseMemObject(value_memobj);
  if (bin_start_memobj) clReleaseMemObject(bin_start_memobj);
  if (sorted_memobj) clReleaseMemObject(sorted_memobj);
  if (griddata_memobj) clReleaseMemObject(griddata_memobj);
  if (density_memobj) clReleaseMemObject(density_memobj);
  if (inter_memobj) clReleaseMemObject(inter_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

static float kernel_value_cpu(float v) {
  const float z = v*v;
  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
              (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
               0.479440257548300e-16f) + 0.435125971262668e-13f ) +
               0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
               0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
               0.463076284721000e0f)   + 0.754337328948189e2f   ) +
               0.830792541809429e4f)   + 0.571661130563785e6f   ) +
               0.216415572361227e8f)   + 0.356644482244025e9f  ) +
               0.144048298227235e10f);
  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);
  return -num/den;
}

static int grid_size = 8;
static int num_samples = 64;
static const int kernel_width = 4;
static const float oversample = 2.0f;

static void show_usage() {
  printf("Usage: [-n grid edge] [-s samples] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:s:h")) != -1) {
    switch (c) {
    case 'n':
      grid_size = atoi(optarg);
      break;
    case 's':
      num_samples = atoi(optarg);
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

// gridding_Gold (CPU_kernels.c): add the contributions of `count` samples.
static void gridding_cpu(int count, const struct ReconstructionSample* sample,
                         int gsize1, int gsize2, int gsize3, float beta,
                         float cutoff, float cutoff2, float one_over_cutoff2,
                         std::vector<float>& gridData,
                         std::vector<float>& sampleDensity) {
  for (int i = 0; i < count; ++i) {
    const struct ReconstructionSample pt = sample[i];
    float kx = pt.kX, ky = pt.kY, kz = pt.kZ;

    int NxL = (int)fmaxf(kx - cutoff, 0.0f);
    int NxH = (int)fminf(kx + cutoff, (float)(gsize1 - 1));
    int NyL = (int)fmaxf(ky - cutoff, 0.0f);
    int NyH = (int)fminf(ky + cutoff, (float)(gsize2 - 1));
    int NzL = (int)fmaxf(kz - cutoff, 0.0f);
    int NzH = (int)fminf(kz + cutoff, (float)(gsize3 - 1));

    if ((pt.real != 0.0f || pt.imag != 0.0f) && pt.sdc != 0.0f) {
      for (int nz = NzL; nz <= NzH; ++nz) {
        float dz2 = (kz - nz) * (kz - nz);
        if (dz2 >= cutoff2)
          continue;
        for (int ny = NyL; ny <= NyH; ++ny) {
          float dy2 = (ky - ny) * (ky - ny);
          float dy2dz2 = dz2 + dy2;
          if (dy2dz2 >= cutoff2)
            continue;
          for (int nx = NxL; nx <= NxH; ++nx) {
            float dx2 = (kx - nx) * (kx - nx);
            float v = dy2dz2 + dx2;
            if (v >= cutoff2)
              continue;
            int idx = nx + ny * gsize1 + nz * gsize1 * gsize2;
            float w = kernel_value_cpu(beta * sqrtf(1.0f - v * one_over_cutoff2)) * pt.sdc;
            gridData[2 * idx] += w * pt.real;
            gridData[2 * idx + 1] += w * pt.imag;
            sampleDensity[idx] += 1.0f;
          }
        }
      }
    }
  }
}

// scanLargeArray.cpp, with a 32-element device block (16 work-items) and the
// dim_block selection extended down to 2 (upstream starts at 128).
static void scan_large_array(cl_mem data, unsigned int gridNumElems) {
  const unsigned int blockSize = 32;
  const unsigned int GRID_SIZE = 65535;
  unsigned int size = (gridNumElems + blockSize - 1) / blockSize;

  unsigned int dim_block = 2;
  unsigned int current_max = size * blockSize;
  for (unsigned int lcv = 2; lcv <= blockSize; lcv *= 2) {
    unsigned int array_size = lcv;
    while (array_size < size)
      array_size *= lcv;
    if (array_size <= current_max) {
      current_max = array_size;
      dim_block = lcv;
    }
  }

  std::vector<unsigned int> zero(current_max, 0);
  inter_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          current_max * sizeof(unsigned int), zero.data(), &_err));

  CL_CHECK(clSetKernelArg(scan_l1_kernel, 1, sizeof(cl_mem), (void *)&data));
  CL_CHECK(clSetKernelArg(scan_l1_kernel, 3, sizeof(cl_mem), (void *)&inter_memobj));
  CL_CHECK(clSetKernelArg(scan_i1_kernel, 0, sizeof(cl_mem), (void *)&inter_memobj));
  CL_CHECK(clSetKernelArg(scan_i2_kernel, 0, sizeof(cl_mem), (void *)&inter_memobj));
  CL_CHECK(clSetKernelArg(uniform_add_kernel, 1, sizeof(cl_mem), (void *)&data));
  CL_CHECK(clSetKernelArg(uniform_add_kernel, 3, sizeof(cl_mem), (void *)&inter_memobj));

  for (unsigned int i = 0; i < (size + GRID_SIZE - 1) / GRID_SIZE; ++i) {
    unsigned int gridSize = ((size - (i * GRID_SIZE)) > GRID_SIZE) ? GRID_SIZE : (size - i * GRID_SIZE);
    unsigned int numElems = gridNumElems - i * GRID_SIZE * blockSize;
    if (numElems > GRID_SIZE * blockSize)
      numElems = GRID_SIZE * blockSize;
    unsigned int data_offset = i * GRID_SIZE * blockSize;
    unsigned int inter_offset = i * GRID_SIZE;
    CL_CHECK(clSetKernelArg(scan_l1_kernel, 0, sizeof(unsigned int), (void *)&numElems));
    CL_CHECK(clSetKernelArg(scan_l1_kernel, 2, sizeof(unsigned int), (void *)&data_offset));
    CL_CHECK(clSetKernelArg(scan_l1_kernel, 4, sizeof(unsigned int), (void *)&inter_offset));
    const size_t block = blockSize / 2;
    const size_t grid = (size_t)gridSize * block;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, scan_l1_kernel, 1, NULL,
                                    &grid, &block, 0, NULL, NULL));
  }

  unsigned int stride = 1;
  for (unsigned int d = current_max; d > 1; d /= dim_block) {
    const size_t block = dim_block / 2;
    const size_t grid = (size_t)(d / dim_block) * block;
    CL_CHECK(clSetKernelArg(scan_i1_kernel, 1, sizeof(unsigned int), (void *)&stride));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, scan_i1_kernel, 1, NULL,
                                    &grid, &block, 0, NULL, NULL));
    stride *= dim_block;
  }

  unsigned int singleZero = 0;
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, inter_memobj, CL_TRUE,
                                (current_max - 1) * sizeof(unsigned int),
                                sizeof(unsigned int), &singleZero, 0, NULL, NULL));

  for (unsigned int d = dim_block; d <= current_max; d *= dim_block) {
    stride /= dim_block;
    const size_t block = dim_block / 2;
    const size_t grid = (size_t)(d / dim_block) * block;
    CL_CHECK(clSetKernelArg(scan_i2_kernel, 1, sizeof(unsigned int), (void *)&stride));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, scan_i2_kernel, 1, NULL,
                                    &grid, &block, 0, NULL, NULL));
  }

  for (unsigned int i = 0; i < (size + GRID_SIZE - 1) / GRID_SIZE; ++i) {
    unsigned int gridSize = ((size - (i * GRID_SIZE)) > GRID_SIZE) ? GRID_SIZE : (size - i * GRID_SIZE);
    unsigned int numElems = gridNumElems - i * GRID_SIZE * blockSize;
    if (numElems > GRID_SIZE * blockSize)
      numElems = GRID_SIZE * blockSize;
    unsigned int data_offset = i * GRID_SIZE * blockSize;
    unsigned int inter_offset = i * GRID_SIZE;
    CL_CHECK(clSetKernelArg(uniform_add_kernel, 0, sizeof(unsigned int), (void *)&numElems));
    CL_CHECK(clSetKernelArg(uniform_add_kernel, 2, sizeof(unsigned int), (void *)&data_offset));
    CL_CHECK(clSetKernelArg(uniform_add_kernel, 4, sizeof(unsigned int), (void *)&inter_offset));
    const size_t block = blockSize / 2;
    const size_t grid = (size_t)gridSize * block;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, uniform_add_kernel, 1, NULL,
                                    &grid, &block, 0, NULL, NULL));
  }
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (grid_size < 4 || num_samples < 1) {
    printf("Error: invalid grid/sample count\n");
    return -1;
  }

  const int gsize1 = grid_size, gsize2 = grid_size, gsize3 = grid_size;
  const int gridNumElems = gsize1 * gsize2 * gsize3;
  const int size_xy = gsize1 * gsize2;
  const int binsize = 16;
  const float cutoff = (float)kernel_width / 2.0f;
  const float cutoff2 = cutoff * cutoff;
  const float one_over_cutoff2 = 1.0f / cutoff2;
  const float ceil_cutoff = ceilf(cutoff);
  const float beta = (float)(3.14159265359 *
      sqrt(4.0 * kernel_width * kernel_width / (oversample * oversample) *
           (oversample - 0.5) * (oversample - 0.5) - 0.8));

  printf("Grid=%dx%dx%d, samples=%d, kernelWidth=%d, binsize=%d, beta=%g\n",
         gsize1, gsize2, gsize3, num_samples, kernel_width, binsize, beta);

  std::vector<struct ReconstructionSample> h_samples(num_samples);
  srand(1);
  for (int i = 0; i < num_samples; ++i) {
    h_samples[i].real = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    h_samples[i].imag = 2.0f * (float)rand() / RAND_MAX - 1.0f;
    // distinct lattice points, at most one sample per bin
    h_samples[i].kX = (float)(i % 4);
    h_samples[i].kY = (float)((i / 4) % 4);
    h_samples[i].kZ = (float)((i / 16) % 4);
    h_samples[i].sdc = 0.5f + (float)rand() / RAND_MAX;
  }

  std::vector<unsigned int> h_keys(num_samples, 0), h_values(num_samples, 0);
  std::vector<unsigned int> h_bin_start(gridNumElems + 1, 0);
  std::vector<struct ReconstructionSample> h_sorted(num_samples);
  std::vector<float> h_grid(2 * (size_t)gridNumElems, 0.0f);
  std::vector<float> h_density(gridNumElems, 0.0f);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  const size_t sample_bytes = (size_t)num_samples * sizeof(struct ReconstructionSample);
  sample_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, sample_bytes, NULL, &_err));
  key_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, num_samples * sizeof(unsigned int), NULL, &_err));
  value_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, num_samples * sizeof(unsigned int), NULL, &_err));
  bin_start_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, (gridNumElems + 1) * sizeof(unsigned int), NULL, &_err));
  sorted_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, sample_bytes, NULL, &_err));
  griddata_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * (size_t)gridNumElems * sizeof(float), NULL, &_err));
  density_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, gridNumElems * sizeof(float), NULL, &_err));

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

  cl_int build_err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (build_err != CL_SUCCESS) {
    char log[8192];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
    printf("Build failed (%d):\n%s\n", (int)build_err, log);
    cleanup();
    return -1;
  }

  binning_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_BINNING, &_err));
  reorder_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_REORDER, &_err));
  grid_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_GRID, &_err));
  scan_l1_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_SCAN_L1, &_err));
  scan_i1_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_SCAN_I1, &_err));
  scan_i2_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_SCAN_I2, &_err));
  uniform_add_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_UNIFORM_ADD, &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, sample_memobj, CL_TRUE, 0, sample_bytes, h_samples.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, bin_start_memobj, CL_TRUE, 0, (gridNumElems + 1) * sizeof(unsigned int), h_bin_start.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, griddata_memobj, CL_TRUE, 0, 2 * (size_t)gridNumElems * sizeof(float), h_grid.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, density_memobj, CL_TRUE, 0, gridNumElems * sizeof(float), h_density.data(), 0, NULL, NULL));

  auto time_start = std::chrono::high_resolution_clock::now();

  printf("Execute binning kernel\n");
  CL_CHECK(clSetKernelArg(binning_kernel, 0, sizeof(unsigned int), (void *)&num_samples));
  CL_CHECK(clSetKernelArg(binning_kernel, 1, sizeof(cl_mem), (void *)&sample_memobj));
  CL_CHECK(clSetKernelArg(binning_kernel, 2, sizeof(cl_mem), (void *)&key_memobj));
  CL_CHECK(clSetKernelArg(binning_kernel, 3, sizeof(cl_mem), (void *)&value_memobj));
  CL_CHECK(clSetKernelArg(binning_kernel, 4, sizeof(cl_mem), (void *)&bin_start_memobj));
  CL_CHECK(clSetKernelArg(binning_kernel, 5, sizeof(int), (void *)&binsize));
  CL_CHECK(clSetKernelArg(binning_kernel, 6, sizeof(unsigned int), (void *)&gridNumElems));
  CL_CHECK(clSetKernelArg(binning_kernel, 7, sizeof(int), (void *)&size_xy));
  CL_CHECK(clSetKernelArg(binning_kernel, 8, sizeof(int), (void *)&gsize1));
  {
    const size_t global = (size_t)((num_samples + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE;
    const size_t local = GROUP_SIZE;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, binning_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));

  printf("Host stable sort of (key,value)\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, key_memobj, CL_TRUE, 0, num_samples * sizeof(unsigned int), h_keys.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, value_memobj, CL_TRUE, 0, num_samples * sizeof(unsigned int), h_values.data(), 0, NULL, NULL));
  {
    std::vector<unsigned int> order(num_samples);
    for (int i = 0; i < num_samples; ++i)
      order[i] = (unsigned int)i;
    std::stable_sort(order.begin(), order.end(),
                     [&](unsigned int a, unsigned int b) { return h_keys[a] < h_keys[b]; });
    std::vector<unsigned int> sorted_values(num_samples);
    for (int i = 0; i < num_samples; ++i)
      sorted_values[i] = h_values[order[i]];
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, value_memobj, CL_TRUE, 0, num_samples * sizeof(unsigned int), sorted_values.data(), 0, NULL, NULL));
  }

  printf("Execute reorder kernel\n");
  CL_CHECK(clSetKernelArg(reorder_kernel, 0, sizeof(int), (void *)&num_samples));
  CL_CHECK(clSetKernelArg(reorder_kernel, 1, sizeof(cl_mem), (void *)&value_memobj));
  CL_CHECK(clSetKernelArg(reorder_kernel, 2, sizeof(cl_mem), (void *)&sample_memobj));
  CL_CHECK(clSetKernelArg(reorder_kernel, 3, sizeof(cl_mem), (void *)&sorted_memobj));
  {
    const size_t global = (size_t)((num_samples + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE;
    const size_t local = GROUP_SIZE;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, reorder_kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL));
  }

  printf("Execute scan\n");
  scan_large_array(bin_start_memobj, gridNumElems + 1);
  CL_CHECK(clFinish(commandQueue));

  unsigned int cpuStart = 0;
  CL_CHECK(clEnqueueReadBuffer(commandQueue, bin_start_memobj, CL_TRUE,
                               gridNumElems * sizeof(unsigned int),
                               sizeof(unsigned int), &cpuStart, 0, NULL, NULL));
  const int cpu_bin_size = num_samples - (int)cpuStart;
  std::vector<struct ReconstructionSample> h_cpu_bin(cpu_bin_size);
  if (cpu_bin_size > 0) {
    CL_CHECK(clEnqueueReadBuffer(commandQueue, sorted_memobj, CL_TRUE,
                                 (size_t)cpuStart * sizeof(struct ReconstructionSample),
                                 (size_t)cpu_bin_size * sizeof(struct ReconstructionSample),
                                 h_cpu_bin.data(), 0, NULL, NULL));
  }

  printf("Execute gridding kernel\n");
  {
    const int dims0 = 4, dims1 = 2, dims2 = 2;
    CL_CHECK(clSetKernelArg(grid_kernel, 0, sizeof(cl_mem), (void *)&sorted_memobj));
    CL_CHECK(clSetKernelArg(grid_kernel, 1, sizeof(cl_mem), (void *)&bin_start_memobj));
    CL_CHECK(clSetKernelArg(grid_kernel, 2, sizeof(cl_mem), (void *)&griddata_memobj));
    CL_CHECK(clSetKernelArg(grid_kernel, 3, sizeof(cl_mem), (void *)&density_memobj));
    CL_CHECK(clSetKernelArg(grid_kernel, 4, sizeof(float), (void *)&beta));
    CL_CHECK(clSetKernelArg(grid_kernel, 5, sizeof(int), (void *)&gsize1));
    CL_CHECK(clSetKernelArg(grid_kernel, 6, sizeof(int), (void *)&gsize2));
    CL_CHECK(clSetKernelArg(grid_kernel, 7, sizeof(int), (void *)&gsize3));
    CL_CHECK(clSetKernelArg(grid_kernel, 8, sizeof(int), (void *)&size_xy));
    CL_CHECK(clSetKernelArg(grid_kernel, 9, sizeof(float), (void *)&cutoff2));
    float ceil_cutoff_f = ceil_cutoff;
    CL_CHECK(clSetKernelArg(grid_kernel, 10, sizeof(float), (void *)&ceil_cutoff_f));
    float one_over = one_over_cutoff2;
    CL_CHECK(clSetKernelArg(grid_kernel, 11, sizeof(float), (void *)&one_over));
    int d0 = dims0, d1 = dims1, d2 = dims2;
    CL_CHECK(clSetKernelArg(grid_kernel, 12, sizeof(int), (void *)&d0));
    CL_CHECK(clSetKernelArg(grid_kernel, 13, sizeof(int), (void *)&d1));
    CL_CHECK(clSetKernelArg(grid_kernel, 14, sizeof(int), (void *)&d2));
    const size_t global[3] = {
      (size_t)(gsize1 / dims0) * dims0,
      (size_t)((gsize2 * gsize3) / (dims1 * dims2)) * dims1,
      (size_t)1 * dims2
    };
    const size_t local[3] = {(size_t)dims0, (size_t)dims1, (size_t)dims2};
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, grid_kernel, 3, NULL,
                                    global, local, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffers\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, griddata_memobj, CL_TRUE, 0,
                               2 * (size_t)gridNumElems * sizeof(float), h_grid.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, density_memobj, CL_TRUE, 0,
                               gridNumElems * sizeof(float), h_density.data(), 0, NULL, NULL));

  printf("Add host-side overflow contributions (%d samples)\n", cpu_bin_size);
  if (cpu_bin_size > 0) {
    gridding_cpu(cpu_bin_size, h_cpu_bin.data(), gsize1, gsize2, gsize3, beta,
                 cutoff, cutoff2, one_over_cutoff2, h_grid, h_density);
  }

  printf("Verify result\n");
  std::vector<float> ref_grid(2 * (size_t)gridNumElems, 0.0f);
  std::vector<float> ref_density(gridNumElems, 0.0f);
  gridding_cpu(num_samples, h_samples.data(), gsize1, gsize2, gsize3, beta,
               cutoff, cutoff2, one_over_cutoff2, ref_grid, ref_density);

  int errors = 0;
  for (int i = 0; i < gridNumElems; ++i) {
    for (int c = 0; c < 2; ++c) {
      float expected = ref_grid[2 * i + c];
      float actual = h_grid[2 * i + c];
      float tol = REL_TOL * fabsf(expected) + ABS_TOL;
      if (fabsf(actual - expected) > tol) {
        if (errors < 100)
          printf("*** error: grid[%d].%s expected=%g, actual=%g (diff=%g)\n",
                 i, c ? "imag" : "real", expected, actual, fabsf(actual - expected));
        ++errors;
      }
    }
    float expected = ref_density[i];
    float actual = h_density[i];
    float tol = REL_TOL * fabsf(expected) + ABS_TOL;
    if (fabsf(actual - expected) > tol) {
      if (errors < 100)
        printf("*** error: density[%d] expected=%g, actual=%g (diff=%g)\n",
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