// Parboil lbm — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/parboil/benchmarks/lbm/src/opencl_base/kernel.cl
//   third_benchmark/parboil/benchmarks/lbm/src/opencl_base/main.c
//     (nTimeSteps loop, grid ping-pong)
//   third_benchmark/parboil/benchmarks/lbm/src/cpu/lbm.c
//     (LBM_initializeGrid, LBM_initializeSpecialCellsForLDC,
//      LBM_performStreamCollide: the host reference used here)
//
// Parboil reads a lattice + obstacle file and compares against a .gold file;
// this port builds the small lid-driven-cavity setup from lbm.c, runs the
// reference stream-and-collide serially and compares the full volume.
//
// Lattice size is compile-time (the kernel's addressing macros need it), so
// this case takes only -t (time steps). The default size is 8x8x8, far below
// Parboil's 120x120x150.
//
// Adaptation: upstream launches dimBlock={SIZE_X,1,1}; Vortex caps a
// work-group at 16, so the kernel maps a flat global id to (x,y,z) and is
// launched with 16-work-item groups. The per-cell math is unchanged.
//
// Data: DFL1/DFL2/DFL3 equilibrium distributions, LDC obstacles/ACCEL
// (lbm.c:120-142), both grids initialised identically. The tolerance is 1e-4
// relative + 1e-6 absolute (the reference uses the kernel's expression order,
// so only FMA contraction can perturb the result).

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "common.h"

#define KERNEL_NAME "performStreamCollide_kernel"

#define LWS 16
#define REL_TOL 1e-4f
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
cl_mem src_memobj = NULL;
cl_mem dst_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (src_memobj) clReleaseMemObject(src_memobj);
  if (dst_memobj) clReleaseMemObject(dst_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int steps = 3;

static void show_usage() {
  printf("Usage: [-t time steps] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "t:h")) != -1) {
    switch (c) {
    case 't':
      steps = atoi(optarg);
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

// lbm.c: LBM_initializeGrid over z in [-2, SIZE_Z+2).
static void init_grid(float* grid) {
  for (int z = -2; z < SIZE_Z + 2; ++z) {
    for (int y = 0; y < SIZE_Y; ++y) {
      for (int x = 0; x < SIZE_X; ++x) {
        float* cell = grid + CALC_INDEX(x, y, z, 0);
        cell[C] = DFL1;
        cell[N] = cell[S] = cell[E] = cell[W] = cell[T] = cell[B] = DFL2;
        cell[NE] = cell[NW] = cell[SE] = cell[SW] = DFL3;
        cell[NT] = cell[NB] = cell[ST] = cell[SB] = DFL3;
        cell[ET] = cell[EB] = cell[WT] = cell[WB] = DFL3;
        cell[FLAGS] = as_float(0u);
      }
    }
  }
}

static void set_flag(float* grid, int x, int y, int z, unsigned int flag) {
  float* cell = grid + CALC_INDEX(x, y, z, 0);
  cell[FLAGS] = as_float(as_uint(cell[FLAGS]) | flag);
}

// lbm.c: LBM_initializeSpecialCellsForLDC.
static void init_special_cells_for_ldc(float* grid) {
  for (int z = -2; z < SIZE_Z + 2; ++z) {
    for (int y = 0; y < SIZE_Y; ++y) {
      for (int x = 0; x < SIZE_X; ++x) {
        if (x == 0 || x == SIZE_X - 1 ||
            y == 0 || y == SIZE_Y - 1 ||
            z == 0 || z == SIZE_Z - 1) {
          set_flag(grid, x, y, z, OBSTACLE);
        } else {
          if ((z == 1 || z == SIZE_Z - 2) &&
              x > 1 && x < SIZE_X - 2 &&
              y > 1 && y < SIZE_Y - 2) {
            set_flag(grid, x, y, z, ACCEL);
          }
        }
      }
    }
  }
}

// lbm.c: LBM_performStreamCollide for one cell. Uses the same macros and the
// same expression order as kernel.cl.
static void stream_collide_cpu(float* srcGrid, float* dstGrid,
                               int sx, int sy, int sz) {
  SWEEP_VAR
  SWEEP_X = sx;
  SWEEP_Y = sy;
  SWEEP_Z = sz;

  float temp_swp, tempC, tempN, tempS, tempE, tempW, tempT, tempB;
  float tempNE, tempNW, tempSE, tempSW, tempNT, tempNB, tempST;
  float tempSB, tempET, tempEB, tempWT, tempWB;

  tempC = SRC_C(srcGrid);

  tempN = SRC_N(srcGrid);
  tempS = SRC_S(srcGrid);
  tempE = SRC_E(srcGrid);
  tempW = SRC_W(srcGrid);
  tempT = SRC_T(srcGrid);
  tempB = SRC_B(srcGrid);

  tempNE = SRC_NE(srcGrid);
  tempNW = SRC_NW(srcGrid);
  tempSE = SRC_SE(srcGrid);
  tempSW = SRC_SW(srcGrid);
  tempNT = SRC_NT(srcGrid);
  tempNB = SRC_NB(srcGrid);
  tempST = SRC_ST(srcGrid);
  tempSB = SRC_SB(srcGrid);
  tempET = SRC_ET(srcGrid);
  tempEB = SRC_EB(srcGrid);
  tempWT = SRC_WT(srcGrid);
  tempWB = SRC_WB(srcGrid);

  if (as_uint(LOCAL(srcGrid, FLAGS)) & (OBSTACLE)) {
    temp_swp = tempN; tempN = tempS; tempS = temp_swp;
    temp_swp = tempE; tempE = tempW; tempW = temp_swp;
    temp_swp = tempT; tempT = tempB; tempB = temp_swp;
    temp_swp = tempNE; tempNE = tempSW; tempSW = temp_swp;
    temp_swp = tempNW; tempNW = tempSE; tempSE = temp_swp;
    temp_swp = tempNT; tempNT = tempSB; tempSB = temp_swp;
    temp_swp = tempNB; tempNB = tempST; tempST = temp_swp;
    temp_swp = tempET; tempET = tempWB; tempWB = temp_swp;
    temp_swp = tempEB; tempEB = tempWT; tempWT = temp_swp;
  } else {
    float ux, uy, uz, rho, u2;
    float temp1, temp2, temp_base;
    rho = tempC + tempN
        + tempS + tempE
        + tempW + tempT
        + tempB + tempNE
        + tempNW + tempSE
        + tempSW + tempNT
        + tempNB + tempST
        + tempSB + tempET
        + tempEB + tempWT
        + tempWB;

    ux = + tempE - tempW
       + tempNE - tempNW
       + tempSE - tempSW
       + tempET + tempEB
       - tempWT - tempWB;

    uy = + tempN - tempS
       + tempNE + tempNW
       - tempSE - tempSW
       + tempNT + tempNB
       - tempST - tempSB;

    uz = + tempT - tempB
       + tempNT - tempNB
       + tempST - tempSB
       + tempET - tempEB
       + tempWT - tempWB;

    ux /= rho;
    uy /= rho;
    uz /= rho;

    if (as_uint(LOCAL(srcGrid, FLAGS)) & (ACCEL)) {
      ux = 0.005f;
      uy = 0.002f;
      uz = 0.000f;
    }

    u2 = 1.5f * (ux*ux + uy*uy + uz*uz) - 1.0f;
    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;

    temp_base = OMEGA*rho;
    temp1 = DFL1*temp_base;
    temp2 = 1.0f-OMEGA;
    tempC = temp2*tempC + temp1*(                                 - u2);
    temp1 = DFL2*temp_base;
    tempN = temp2*tempN + temp1*(       uy*(4.5f*uy       + 3.0f) - u2);
    tempS = temp2*tempS + temp1*(       uy*(4.5f*uy       - 3.0f) - u2);
    tempT = temp2*tempT + temp1*(       uz*(4.5f*uz       + 3.0f) - u2);
    tempB = temp2*tempB + temp1*(       uz*(4.5f*uz       - 3.0f) - u2);
    tempE = temp2*tempE + temp1*(       ux*(4.5f*ux       + 3.0f) - u2);
    tempW = temp2*tempW + temp1*(       ux*(4.5f*ux       - 3.0f) - u2);
    temp1 = DFL3*temp_base;
    tempNT= temp2*tempNT + temp1 *( (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
    tempNB= temp2*tempNB + temp1 *( (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
    tempST= temp2*tempST + temp1 *( (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
    tempSB= temp2*tempSB + temp1 *( (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
    tempNE = temp2*tempNE + temp1 *( (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
    tempSE = temp2*tempSE + temp1 *((+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
    tempET = temp2*tempET + temp1 *( (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
    tempEB = temp2*tempEB + temp1 *( (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
    tempNW = temp2*tempNW + temp1 *( (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
    tempSW = temp2*tempSW + temp1 *( (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
    tempWT = temp2*tempWT + temp1 *( (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
    tempWB = temp2*tempWB + temp1 *( (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
  }

  DST_C (dstGrid) = tempC;

  DST_N (dstGrid) = tempN;
  DST_S (dstGrid) = tempS;
  DST_E (dstGrid) = tempE;
  DST_W (dstGrid) = tempW;
  DST_T (dstGrid) = tempT;
  DST_B (dstGrid) = tempB;

  DST_NE(dstGrid) = tempNE;
  DST_NW(dstGrid) = tempNW;
  DST_SE(dstGrid) = tempSE;
  DST_SW(dstGrid) = tempSW;
  DST_NT(dstGrid) = tempNT;
  DST_NB(dstGrid) = tempNB;
  DST_ST(dstGrid) = tempST;
  DST_SB(dstGrid) = tempSB;
  DST_ET(dstGrid) = tempET;
  DST_EB(dstGrid) = tempEB;
  DST_WT(dstGrid) = tempWT;
  DST_WB(dstGrid) = tempWB;
}

static void stream_collide_volume_cpu(float* srcGrid, float* dstGrid) {
  for (int z = 0; z < SIZE_Z; ++z)
    for (int y = 0; y < SIZE_Y; ++y)
      for (int x = 0; x < SIZE_X; ++x)
        stream_collide_cpu(srcGrid, dstGrid, x, y, z);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (steps < 1) {
    printf("Error: steps must be at least 1\n");
    return -1;
  }

  printf("Lattice size=%dx%dx%d, time steps=%d\n", SIZE_X, SIZE_Y, SIZE_Z, steps);

  const size_t grid_size = (size_t)TOTAL_PADDED_CELLS * N_CELL_ENTRIES;
  const size_t grid_nbytes = grid_size * sizeof(float);

  std::vector<float> h_a(grid_size), h_b(grid_size);
  float* a = h_a.data() + MARGIN;
  float* b = h_b.data() + MARGIN;
  init_grid(a);
  init_special_cells_for_ldc(a);
  init_grid(b);
  init_special_cells_for_ldc(b);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  src_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, grid_nbytes, NULL, &_err));
  dst_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, grid_nbytes, NULL, &_err));

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
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, src_memobj, CL_TRUE, 0, grid_nbytes, h_a.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, dst_memobj, CL_TRUE, 0, grid_nbytes, h_b.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const size_t global_work_size = (size_t)TOTAL_CELLS;
  const size_t local_work_size = LWS;
  auto time_start = std::chrono::high_resolution_clock::now();
  int src_index = 0;
  // main.c:50-54: kernel(src,dst), then swap.
  for (int t = 0; t < steps; ++t) {
    cl_mem d_src = src_index == 0 ? src_memobj : dst_memobj;
    cl_mem d_dst = src_index == 0 ? dst_memobj : src_memobj;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_src));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_dst));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    &global_work_size, &local_work_size,
                                    0, NULL, NULL));
    src_index ^= 1;
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  cl_mem d_result = src_index == 0 ? src_memobj : dst_memobj;
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_result, CL_TRUE, 0, grid_nbytes,
                               h_a.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Replay the same ping-pong on the host.
  std::vector<float> r_a(grid_size), r_b(grid_size);
  float* ra = r_a.data() + MARGIN;
  float* rb = r_b.data() + MARGIN;
  init_grid(ra);
  init_special_cells_for_ldc(ra);
  init_grid(rb);
  init_special_cells_for_ldc(rb);
  float* cur_src = ra;
  float* cur_dst = rb;
  for (int t = 0; t < steps; ++t) {
    stream_collide_volume_cpu(cur_src, cur_dst);
    std::swap(cur_src, cur_dst);
  }

  int errors = 0;
  for (int z = 0; z < SIZE_Z; ++z) {
    for (int y = 0; y < SIZE_Y; ++y) {
      for (int x = 0; x < SIZE_X; ++x) {
        const float* ref_cell = cur_src + CALC_INDEX(x, y, z, 0);
        const float* out_cell = a + CALC_INDEX(x, y, z, 0);
        for (int e = C; e < FLAGS; ++e) {
          float expected = ref_cell[e];
          float actual = out_cell[e];
          float tol = REL_TOL * fabsf(expected) + ABS_TOL;
          if (fabsf(actual - expected) > tol) {
            if (errors < 100) {
              printf("*** error: [%d,%d,%d,%d] expected=%f, actual=%f (diff=%g)\n",
                     x, y, z, e, expected, actual, fabsf(actual - expected));
            }
            ++errors;
          }
        }
        if (as_uint(ref_cell[FLAGS]) != as_uint(out_cell[FLAGS])) {
          if (errors < 100) {
            printf("*** error: [%d,%d,%d] flags expected=%u, actual=%u\n",
                   x, y, z, as_uint(ref_cell[FLAGS]), as_uint(out_cell[FLAGS]));
          }
          ++errors;
        }
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