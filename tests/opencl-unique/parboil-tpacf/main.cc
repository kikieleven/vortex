// Parboil tpacf — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/parboil/benchmarks/tpacf/src/opencl_base/kernel.cl
//   third_benchmark/parboil/benchmarks/tpacf/src/opencl_base/main.cc
//     (initBinB: bin boundaries; set/sample launch of (2*NUM_SETS+1) blocks)
//   third_benchmark/parboil/benchmarks/tpacf/src/opencl_base/model.cc
//     (spherical -> cartesian conversion; this port generates random unit
//      vectors directly)
//
// Parboil reads a data file plus NUM_SETS random-point files and writes the
// histograms; this port generates deterministic random unit vectors and
// compares the full histogram array against a host reference that follows the
// kernel's pair-counting rules.
//
// Adaptations (see kernel.cl): BLOCK_SIZE/WARP_SIZE are 16 so the local
// warp-histogram reduction is self-consistent at the 16-work-item group cap;
// atom_inc -> atomic_inc; the final histogram store loops over bins. Atomics
// need the Vortex A extension, so this case is in EXCLUDE and runs via its
// opencl.yaml blackbox entry.
//
// Histograms are integer counts, so the comparison is exact.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "common.h"

#define KERNEL_NAME "gen_hists"

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
cl_mem hist_memobj = NULL;
cl_mem data_memobj = NULL;
cl_mem binb_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (hist_memobj) clReleaseMemObject(hist_memobj);
  if (data_memobj) clReleaseMemObject(data_memobj);
  if (binb_memobj) clReleaseMemObject(binb_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int num_elements = 64;
int num_sets = 2;

static void show_usage() {
  printf("Usage: [-n points per set] [-s random sets] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:s:h")) != -1) {
    switch (c) {
    case 'n':
      num_elements = atoi(optarg);
      break;
    case 's':
      num_sets = atoi(optarg);
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

static unsigned int bin_search(float distance, const float* binb) {
  unsigned int min = 0;
  unsigned int max = NUM_BINS;
  while (max > min + 1) {
    unsigned int k2 = (min + max) / 2;
    if (distance >= binb[k2])
      max = k2;
    else
      min = k2;
  }
  return max - 1;
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (num_elements < 1 || num_sets < 0) {
    printf("Error: invalid sizes\n");
    return -1;
  }

  const int total_sets = num_sets + 1;
  const int num_blocks = num_sets * 2 + 1;
  printf("Elements per set=%d, random sets=%d, blocks=%d\n",
         num_elements, num_sets, num_blocks);

  // initBinB (main.cc)
  std::vector<float> h_binb(NUM_BINS + 1);
  for (int k = 0; k < NUM_BINS + 1; ++k) {
    h_binb[k] = (float)cos(pow(10.0, (log10(min_arcmin) + k * 1.0 / bins_per_dec))
                           / 60.0 * D2R);
  }

  // (1 + NUM_SETS) sets of random unit vectors, stored as SoA x|y|z
  const size_t set_size = (size_t)num_elements;
  const size_t soa_size = (size_t)total_sets * set_size;
  std::vector<float> h_x(soa_size), h_y(soa_size), h_z(soa_size);
  srand(1);
  for (int s = 0; s < total_sets; ++s) {
    for (int i = 0; i < num_elements; ++i) {
      float x = 2.0f * (float)rand() / RAND_MAX - 1.0f;
      float y = 2.0f * (float)rand() / RAND_MAX - 1.0f;
      float z = 2.0f * (float)rand() / RAND_MAX - 1.0f;
      float len = sqrtf(x*x + y*y + z*z);
      if (len == 0.0f) { x = 1.0f; len = 1.0f; }
      size_t idx = (size_t)s * set_size + i;
      h_x[idx] = x / len;
      h_y[idx] = y / len;
      h_z[idx] = z / len;
    }
  }

  std::vector<float> h_data(3 * soa_size);
  for (size_t i = 0; i < soa_size; ++i) {
    h_data[i] = h_x[i];
    h_data[soa_size + i] = h_y[i];
    h_data[2 * soa_size + i] = h_z[i];
  }
  std::vector<unsigned long> h_hists((size_t)NUM_BINS * num_blocks, 0);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  hist_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_hists.size() * sizeof(unsigned long), NULL, &_err));
  data_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, h_data.size() * sizeof(float), NULL, &_err));
  binb_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, h_binb.size() * sizeof(float), NULL, &_err));

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

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, hist_memobj, CL_TRUE, 0, h_hists.size() * sizeof(unsigned long), h_hists.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, data_memobj, CL_TRUE, 0, h_data.size() * sizeof(float), h_data.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, binb_memobj, CL_TRUE, 0, h_binb.size() * sizeof(float), h_binb.data(), 0, NULL, NULL));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&hist_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&data_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&binb_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void *)&num_sets));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void *)&num_elements));

  printf("Execute the kernel\n");
  const size_t global_work_size = (size_t)num_blocks * BLOCK_SIZE;
  const size_t local_work_size = BLOCK_SIZE;
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, hist_memobj, CL_TRUE, 0,
                               h_hists.size() * sizeof(unsigned long),
                               h_hists.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Host reference: same set schedule and pair filter as the kernel.
  std::vector<unsigned long> ref_hists((size_t)NUM_BINS * num_blocks, 0);
  for (int bx = 0; bx < num_blocks; ++bx) {
    const bool do_self = bx < (num_sets + 1);
    const int data_set = do_self ? bx : 0;
    const int rand_set = do_self ? bx : bx - num_sets;

    for (int i = 0; i < num_elements; ++i) {
      const float rx = h_x[(size_t)rand_set * set_size + i];
      const float ry = h_y[(size_t)rand_set * set_size + i];
      const float rz = h_z[(size_t)rand_set * set_size + i];

      for (int k = 0; k < num_elements; ++k) {
        if (do_self && !(i > k))
          continue;
        const float dx = h_x[(size_t)data_set * set_size + k];
        const float dy = h_y[(size_t)data_set * set_size + k];
        const float dz = h_z[(size_t)data_set * set_size + k];

        float distance = dx * rx + dy * ry + dz * rz;
        if (!(distance < h_binb[0]))
          continue;  // bin search needs min > 0; such pairs are never counted

        unsigned int bin = bin_search(distance, h_binb.data());
        // the kernel's explicit guard (reproduced verbatim)
        unsigned int min = 0, max = NUM_BINS;
        {
          unsigned int k2;
          while (max > min + 1) {
            k2 = (min + max) / 2;
            if (distance >= h_binb[k2]) max = k2;
            else min = k2;
          }
        }
        if ((distance < h_binb[min]) && (distance >= h_binb[max]) &&
            (!do_self || (i > k)) && (i < num_elements))
          ref_hists[(size_t)bx * NUM_BINS + bin]++;
      }
    }
  }

  int errors = 0;
  for (size_t i = 0; i < h_hists.size(); ++i) {
    if (ref_hists[i] != h_hists[i]) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%lu, actual=%lu\n",
               (int)i, ref_hists[i], h_hists[i]);
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