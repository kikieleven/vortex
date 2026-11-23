// Parboil cutcp — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/parboil/benchmarks/cutcp/src/opencl_base/kernel.cl
//   third_benchmark/parboil/benchmarks/cutcp/src/opencl_base/cutoff.c
//     (binning, neighbor list, per-z-region launch, region layout)
//   third_benchmark/parboil/benchmarks/cutcp/src/opencl_base/cutcpu.c
//     (direct cutoff-potential sum, used as the host reference)
//
// Parboil reads a PQR file and writes a lattice summary; this port generates a
// deterministic atom cloud, builds the same bins/neighbor list, and compares
// the lattice against the direct pair-potential sum.
//
// Adaptations (see kernel.cl): 16-work-item groups iterating the upstream
// 128 logical threads, flat int/float arrays in place of int4/float4, scalar
// NbrListLen, explicit xRegionDim/yRegionDim arguments.
//
// Data: 16 atoms in a 6 A cube (h = 0.5 A, cutoff = 1.0 A, one bin of margin),
// charges of both signs. The upstream exclusion pass (cutoff 1.0 A) is applied
// on the host by skipping lattice points within exclcutoff of any atom; those
// points are not part of the comparison. The tolerance is 1e-3 relative +
// 1e-3 absolute (the accumulator order differs between bins and direct sum).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "common.h"

#define KERNEL_NAME "opencl_cutoff_potential_lattice"

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
cl_mem bin_memobj = NULL;
cl_mem region_memobj = NULL;
cl_mem nbrlist_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (bin_memobj) clReleaseMemObject(bin_memobj);
  if (region_memobj) clReleaseMemObject(region_memobj);
  if (nbrlist_memobj) clReleaseMemObject(nbrlist_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 6;      // domain edge length in Angstroms
int natoms = 16;

static void show_usage() {
  printf("Usage: [-n domain edge length] [-a atom count] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:a:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'a':
      natoms = atoi(optarg);
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

struct Atom {
  float x, y, z, q;
};

// Region layout produced by the kernel:
//   region (x/8, y/8, z/8), sub-region (z%8)/2, then lz*64 + ly*8 + lx.
static size_t region_index(int x, int y, int z,
                           int xRegionDim, int yRegionDim) {
  int xr = x / 8, yr = y / 8, zr = z / 8;
  int sub = (z % 8) / 2;
  int lz = z % 2, ly = y % 8, lx = x % 8;
  return ((size_t)((zr * yRegionDim + yr) * xRegionDim + xr)) * REGION_SIZE
       + sub * SUB_REGION_SIZE + (lz * 64 + ly * 8 + lx);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 4 || natoms < 1) {
    printf("Error: invalid domain/atom count\n");
    return -1;
  }

  const float h = 0.5f;          // main.c:97
  const float cutoff = 1.0f;
  const float exclcutoff = 1.0f; // main.c:99
  const float lo[3] = {0.0f, 0.0f, 0.0f};
  const int nx = (int)floorf((size - lo[0]) / h) + 1;
  const int ny = nx, nz = nx;

  const int xRegionDim = (nx + 7) / 8;
  const int yRegionDim = (ny + 7) / 8;
  const int zRegionDim = (nz + 7) / 8;
  const int lnx = 8 * xRegionDim;
  const int lny = 8 * yRegionDim;
  const int lnz = 8 * zRegionDim;
  const int lnall = lnx * lny * lnz;

  const int c = (int)ceilf(cutoff * BIN_INVLEN);
  const int binDim_x = (int)ceilf(lnx * h * BIN_INVLEN) + 2 * c;
  const int binDim_y = (int)ceilf(lny * h * BIN_INVLEN) + 2 * c;
  const int binDim_z = (int)ceilf(lnz * h * BIN_INVLEN) + 2 * c;
  const int nbins = binDim_x * binDim_y * binDim_z;
  const int offset = ((c * binDim_y + c) * binDim_x + c) * BIN_SIZE;

  printf("Domain=%d A, atoms=%d, lattice=%dx%dx%d (padded %dx%dx%d)\n",
         size, natoms, nx, ny, nz, lnx, lny, lnz);
  printf("bins=%dx%dx%d, cutoff=%g A, h=%g A\n", binDim_x, binDim_y, binDim_z, cutoff, h);

  // deterministic atom cloud inside [0.5, size-0.5]^3
  std::vector<struct Atom> atoms(natoms);
  srand(7);
  for (int n = 0; n < natoms; ++n) {
    atoms[n].x = 0.5f + (size - 1.0f) * (float)rand() / RAND_MAX;
    atoms[n].y = 0.5f + (size - 1.0f) * (float)rand() / RAND_MAX;
    atoms[n].z = 0.5f + (size - 1.0f) * (float)rand() / RAND_MAX;
    float mag = 0.5f + (float)rand() / RAND_MAX;
    atoms[n].q = (rand() & 1) ? mag : -mag;
  }

  // geometric hashing into bins (cutoff.c:157-198)
  std::vector<float> h_bins((size_t)nbins * BIN_SIZE, 0.0f);
  std::vector<int> h_bincnt(nbins, 0);
  float* binZero = h_bins.data() + offset;
  int excluded_atoms = 0;
  std::vector<bool> in_bin(natoms, false);
  for (int n = 0; n < natoms; ++n) {
    const struct Atom& a = atoms[n];
    float px = a.x - lo[0];
    float py = a.y - lo[1];
    float pz = a.z - lo[2];
    int i = (int)floorf(px * BIN_INVLEN);
    int j = (int)floorf(py * BIN_INVLEN);
    int k = (int)floorf(pz * BIN_INVLEN);
    if (i >= -c && i < binDim_x - c &&
        j >= -c && j < binDim_y - c &&
        k >= -c && k < binDim_z - c &&
        a.q != 0.0f) {
      in_bin[n] = true;
      int index = (k * binDim_y + j) * binDim_x + i;
      float* bin = binZero + (size_t)index * BIN_SIZE;
      int bindex = h_bincnt[index + (c * binDim_y + c) * binDim_x + c];
      if (bindex >= BIN_DEPTH) {
        printf("Error: atom bin overflow (increase the domain or lower --atoms)\n");
        return -1;
      }
      bin[bindex * 4 + 0] = px;
      bin[bindex * 4 + 1] = py;
      bin[bindex * 4 + 2] = pz;
      bin[bindex * 4 + 3] = a.q;
      // counters use the shifted origin so a bin index can be negative
      h_bincnt[index + (c * binDim_y + c) * binDim_x + c] = bindex + 1;
    } else {
      ++excluded_atoms;
    }
  }

  // neighbor list (cutoff.c:105-150): bin offsets within the cutoff plus one
  // bin of slack
  std::vector<int> h_nbrlist;
  {
    const float s = sqrtf(3.0f);
    const float r2 = (cutoff + s * BIN_LENGTH) * (cutoff + s * BIN_LENGTH);
    for (int k = -c; k <= c; ++k)
      for (int j = -c; j <= c; ++j)
        for (int i = -c; i <= c; ++i) {
          if ((i*i + j*j + k*k) * BIN_LENGTH * BIN_LENGTH >= r2)
            continue;
          h_nbrlist.push_back(i);
          h_nbrlist.push_back(j);
          h_nbrlist.push_back(k);
        }
  }
  const int nbrlistlen = (int)(h_nbrlist.size() / 3);

  std::vector<float> h_region(lnall, 0.0f);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  bin_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, h_bins.size() * sizeof(float), NULL, &_err));
  region_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, lnall * sizeof(float), NULL, &_err));
  nbrlist_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, h_nbrlist.size() * sizeof(int), NULL, &_err));

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
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, bin_memobj, CL_TRUE, 0, h_bins.size() * sizeof(float), h_bins.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, region_memobj, CL_TRUE, 0, lnall * sizeof(float), h_region.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, nbrlist_memobj, CL_TRUE, 0, h_nbrlist.size() * sizeof(int), h_nbrlist.data(), 0, NULL, NULL));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int), (void *)&binDim_x));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), (void *)&binDim_y));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bin_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), (void *)&offset));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float), (void *)&h));
  float cutoff2 = cutoff * cutoff;
  float inv_cutoff2 = 1.0f / cutoff2;
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float), (void *)&cutoff2));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(float), (void *)&inv_cutoff2));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&region_memobj));
  CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), (void *)&nbrlistlen));
  CL_CHECK(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void *)&nbrlist_memobj));
  CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int), (void *)&xRegionDim));
  CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int), (void *)&yRegionDim));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  for (int zRegionIndex = 0; zRegionIndex < zRegionDim; ++zRegionIndex) {
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), (void *)&zRegionIndex));
    const size_t global = (size_t)(4 * xRegionDim * yRegionDim) * GROUP_SIZE;
    const size_t local = GROUP_SIZE;
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    &global, &local, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, region_memobj, CL_TRUE, 0, lnall * sizeof(float),
                               h_region.data(), 0, NULL, NULL));

  printf("Verify result\n");
  const float cutoff2_ref = cutoff * cutoff;
  const float inv_cutoff2_ref = 1.0f / cutoff2_ref;
  int errors = 0;
  int compared = 0;
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        float px = i * h, py = j * h, pz = k * h;

        // skip the points the upstream exclusion pass zeroes
        bool excluded = false;
        for (int n = 0; n < natoms; ++n) {
          if (!in_bin[n])
            continue;
          float dx = atoms[n].x - px;
          float dy = atoms[n].y - py;
          float dz = atoms[n].z - pz;
          if (dx*dx + dy*dy + dz*dz < exclcutoff * exclcutoff) {
            excluded = true;
            break;
          }
        }
        if (excluded)
          continue;

        float expected = 0.0f;
        for (int n = 0; n < natoms; ++n) {
          if (!in_bin[n])
            continue;
          float dx = atoms[n].x - px;
          float dy = atoms[n].y - py;
          float dz = atoms[n].z - pz;
          float r2 = dx*dx + dy*dy + dz*dz;
          if (r2 < cutoff2_ref) {
            float s = 1.0f - r2 * inv_cutoff2_ref;
            expected += atoms[n].q * (1.0f / sqrtf(r2)) * s * s;
          }
        }

        float actual = h_region[region_index(i, j, k, xRegionDim, yRegionDim)];
        float tol = REL_TOL * fabsf(expected) + ABS_TOL;
        if (fabsf(actual - expected) > tol) {
          if (errors < 100) {
            printf("*** error: [%d,%d,%d] expected=%g, actual=%g (diff=%g)\n",
                   i, j, k, expected, actual, fabsf(actual - expected));
          }
          ++errors;
        }
        ++compared;
      }
    }
  }

  if (excluded_atoms != 0)
    printf("Note: %d atoms outside the binned domain\n", excluded_atoms);
  printf("Compared %d lattice points\n", compared);

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}