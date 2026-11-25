// SHOC MD (level1/md) — standalone self-checking OpenCL port for Vortex,
// single precision only.
//
// Source:
//   third_benchmark/shoc/src/opencl/level1/md/md.cl            (kernel)
//   third_benchmark/shoc/src/opencl/level1/md/MD.cpp:210-410   (setup/launch)
//   third_benchmark/shoc/src/opencl/level1/md/MD.cpp:59-105    (checkResults)
//   third_benchmark/shoc/src/opencl/level1/md/MD.cpp:499-700   (distance /
//                                                              neighbor list)
//
// SHOC runs this kernel in single precision unconditionally and then in double
// precision if the device advertises cl_khr_fp64 (MD.cpp:296-330). Vortex has
// no fp64 support and this port is single-precision only, so the DP path is
// not implemented and the SINGLE_PRECISION macros are replaced by float.
//
// Fidelity notes:
//   - positions are generated with the same srand48(8650341) / drand48()
//     sequence inside the [0, domainEdge) cube (MD.cpp:282-290);
//   - the 128-entry nearest-neighbour list is built with the same
//     insertInOrder/populateNeighborList algorithm and the same
//     [j*nAtom + i] layout, because neighbour order changes the float
//     accumulation order;
//   - the CPU reference evaluates the same expression in the same order
//     (md.cl compute_lj_force vs MD.cpp checkResults).
//
// Comparison: |actual - expected| <= REL_TOL*|expected| + ABS_TOL per force
// component. SHOC's checkResults divides by the device result and uses a 0.3
// relative threshold (3*EPSILON); this port keeps the same arithmetic on both
// sides but uses a tighter, safer mixed tolerance that does not divide by a
// possibly-zero component.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <list>
#include <vector>

#define KERNEL_NAME "compute_lj_force"

// Vortex's OpenCL device caps a work-group at 16 work-items.
#define LWS 16

// SHOC MD.h constants.
static const float  CUTSQ        = 16.0f;
static const int    MAX_NEIGHBORS = 128;
static const double DOMAIN_EDGE  = 20.0;
static const float  LJ1          = 1.5f;
static const float  LJ2          = 2.0f;

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

// Host mirror of OpenCL float4 (same 16-byte layout).
struct Vec4 {
  float x, y, z, w;
};

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

// SHOC insertInOrder (MD.cpp:534).
static void insert_in_order(std::list<float>& curr_dist,
                            std::list<int>& curr_list,
                            int j, float dist_ij) {
  if (dist_ij > curr_dist.back())
    return;

  std::list<int>::iterator it2 = curr_list.begin();
  for (std::list<float>::iterator it = curr_dist.begin();
       it != curr_dist.end(); ++it, ++it2) {
    if (dist_ij < *it) {
      curr_dist.insert(it, dist_ij);
      curr_list.insert(it2, j);
      curr_list.resize(MAX_NEIGHBORS);
      curr_dist.resize(MAX_NEIGHBORS);
      return;
    }
  }
}

// SHOC buildNeighborList (MD.cpp:586).
static void build_neighbor_list(const std::vector<Vec4>& position,
                                std::vector<int>& neighbor_list) {
  const int n = (int)position.size();
  neighbor_list.assign((size_t)n * MAX_NEIGHBORS, -1);

  for (int i = 0; i < n; ++i) {
    std::list<int> curr_list(MAX_NEIGHBORS, -1);
    std::list<float> curr_dist(MAX_NEIGHBORS, FLT_MAX);

    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      float delx = position[i].x - position[j].x;
      float dely = position[i].y - position[j].y;
      float delz = position[i].z - position[j].z;
      float dist_ij = delx * delx + dely * dely + delz * delz;
      insert_in_order(curr_dist, curr_list, j, dist_ij);
    }

    int idx = 0;
    for (std::list<int>::iterator it = curr_list.begin();
         it != curr_list.end(); ++it, ++idx) {
      neighbor_list[(size_t)idx * n + i] = *it;
    }
  }
}

// SHOC checkResults (MD.cpp:59), same force expression as the kernel.
static void lj_force_cpu(const std::vector<Vec4>& position,
                         const std::vector<int>& neighbor_list,
                         std::vector<Vec4>& force) {
  const int n = (int)position.size();
  force.assign(n, Vec4{0.0f, 0.0f, 0.0f, 0.0f});

  for (int i = 0; i < n; ++i) {
    Vec4 ipos = position[i];
    Vec4 f = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < MAX_NEIGHBORS; ++j) {
      int jidx = neighbor_list[(size_t)j * n + i];
      Vec4 jpos = position[jidx];
      float delx = ipos.x - jpos.x;
      float dely = ipos.y - jpos.y;
      float delz = ipos.z - jpos.z;
      float r2inv = delx * delx + dely * dely + delz * delz;
      if (r2inv < CUTSQ) {
        r2inv = 1.0f / r2inv;
        float r6inv = r2inv * r2inv * r2inv;
        float force_c = r2inv * r6inv * (LJ1 * r6inv - LJ2);
        f.x += delx * force_c;
        f.y += dely * force_c;
        f.z += delz * force_c;
      }
    }
    force[i] = f;
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem force_memobj = NULL;
cl_mem position_memobj = NULL;
cl_mem neighbor_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (force_memobj) clReleaseMemObject(force_memobj);
  if (position_memobj) clReleaseMemObject(position_memobj);
  if (neighbor_memobj) clReleaseMemObject(neighbor_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int natoms = 1024;

static void show_usage() {
  printf("Usage: [-n atoms] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
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

  printf("float workload atoms=%d\n", natoms);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (natoms <= MAX_NEIGHBORS) {
    printf("Error: atoms must be greater than %d to fill the neighbour list\n",
           MAX_NEIGHBORS);
    return -1;
  }

  const int n = natoms;
  const size_t pos_nbytes = (size_t)n * sizeof(Vec4);
  const size_t force_nbytes = (size_t)n * sizeof(Vec4);
  const size_t neigh_nbytes = (size_t)n * MAX_NEIGHBORS * sizeof(int);

  // SHOC MD.cpp:282-290: fixed seed, [0, domainEdge) cube.
  std::vector<Vec4> h_position(n);
  srand48(8650341L);
  for (int i = 0; i < n; ++i) {
    h_position[i].x = (float)(drand48() * DOMAIN_EDGE);
    h_position[i].y = (float)(drand48() * DOMAIN_EDGE);
    h_position[i].z = (float)(drand48() * DOMAIN_EDGE);
    h_position[i].w = 0.0f;
  }

  printf("Building neighbour list (O(n^2))\n");
  std::vector<int> h_neighbor_list;
  build_neighbor_list(h_position, h_neighbor_list);

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

  force_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          force_nbytes, NULL, &_err));
  position_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                             pos_nbytes, NULL, &_err));
  neighbor_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                             neigh_nbytes, NULL, &_err));

  int neigh_count = MAX_NEIGHBORS;
  float cutsq = CUTSQ;
  float lj1 = LJ1;
  float lj2 = LJ2;
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&force_memobj));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&position_memobj));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), (void *)&neigh_count));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&neighbor_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(float), (void *)&cutsq));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(float), (void *)&lj1));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(float), (void *)&lj2));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), (void *)&n));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, position_memobj, CL_TRUE, 0,
                                pos_nbytes, h_position.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, neighbor_memobj, CL_TRUE, 0,
                                neigh_nbytes, h_neighbor_list.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t local_work_size = LWS;
  const size_t global_work_size = ((size_t)n + LWS - 1) / LWS * LWS;
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  std::vector<Vec4> h_force(n);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, force_memobj, CL_TRUE, 0,
                               force_nbytes, h_force.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<Vec4> h_ref;
  lj_force_cpu(h_position, h_neighbor_list, h_ref);

  int errors = 0;
  for (int i = 0; i < n; ++i) {
    const float* act = &h_force[i].x;
    const float* exp = &h_ref[i].x;
    for (int c = 0; c < 3; ++c) {
      float tol = REL_TOL * fabsf(exp[c]) + ABS_TOL;
      if (fabsf(act[c] - exp[c]) > tol) {
        if (errors < 100) {
          printf("*** error: atom %d component %d expected=%f, actual=%f (diff=%g)\n",
                 i, c, exp[c], act[c], fabsf(act[c] - exp[c]));
        }
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
