// Leukocyte (Rodinia) — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/gpu-rodinia/opencl/leukocyte/OpenCL
//   find_ellipse_kernel.cl  -> GICOV_kernel, dilate_kernel
//   track_ellipse_kernel.cl -> IMGVF_kernel
//
// Scope (same reduction as the existing heartwall port): only the three GPU
// kernels are covered. The upstream AVI reader, meschach matrix library,
// active-contour snake and the cell detection/tracking drivers are host-side
// scaffolding and are not ported. Inputs are synthetic and deterministic:
//   * a smooth bright ellipse on a faint sinusoidal background (gradients are
//     computed on the host with the same central-difference stencil as
//     upstream misc_math.c gradient_x/gradient_y);
//   * one 21x41 cell image for the MGVF kernel, normalized to [0,1] exactly
//     like upstream MGVF().
//
// Vortex mapping: work-groups of 16. The upstream IMGVF kernel pipelines the
// update through local buffers; this port uses a clean Jacobi double buffer
// (all reads from the previous iteration), which is the intended MGVF math and
// makes the CPU reference deterministic. The convergence early-exit is not
// exercised: the reference runs a fixed number of iterations.
//
// Every stage (GICOV, dilate, IMGVF) is compared against a serial CPU
// reference using the same single-precision arithmetic.

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
cl_kernel gicov_kernel = NULL, dilate_kernel = NULL, imgvf_kernel = NULL;
cl_mem d_gx = NULL, d_gy = NULL, d_sin = NULL, d_cos = NULL;
cl_mem d_tX = NULL, d_tY = NULL, d_gicov = NULL;
cl_mem d_strel = NULL, d_dilated = NULL;
cl_mem d_cell = NULL, d_imgvf = NULL, d_offsets = NULL, d_m = NULL, d_n = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (gicov_kernel) clReleaseKernel(gicov_kernel);
  if (dilate_kernel) clReleaseKernel(dilate_kernel);
  if (imgvf_kernel) clReleaseKernel(imgvf_kernel);
  if (program) clReleaseProgram(program);
  if (d_gx) clReleaseMemObject(d_gx);
  if (d_gy) clReleaseMemObject(d_gy);
  if (d_sin) clReleaseMemObject(d_sin);
  if (d_cos) clReleaseMemObject(d_cos);
  if (d_tX) clReleaseMemObject(d_tX);
  if (d_tY) clReleaseMemObject(d_tY);
  if (d_gicov) clReleaseMemObject(d_gicov);
  if (d_strel) clReleaseMemObject(d_strel);
  if (d_dilated) clReleaseMemObject(d_dilated);
  if (d_cell) clReleaseMemObject(d_cell);
  if (d_imgvf) clReleaseMemObject(d_imgvf);
  if (d_offsets) clReleaseMemObject(d_offsets);
  if (d_m) clReleaseMemObject(d_m);
  if (d_n) clReleaseMemObject(d_n);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// ---------------------------------------------------------------------------
// Upstream constants (find_ellipse.c / find_ellipse_kernel.cl /
// track_ellipse_kernel.cl)
// ---------------------------------------------------------------------------
#define NPOINTS 150
#define NCIRCLES 7
#define MIN_RAD 8
#define MAX_RAD 20
#define MAXR (MAX_RAD + 2)
#define STREL_RADIUS 12
#define STREL_SIZE (STREL_RADIUS * 2 + 1)
#define M_CELL 21
#define N_CELL 41
#define LOCAL_SIZE 16
static const double PI_D = 3.14159265358979323846;
static const float PI_F = 3.14159f;

static int FrameSize = 48;
static int MgvfIters = 20;

static void show_usage() {
  printf("Usage: [-s frame_size] [-t mgvf_iterations] [-h]\n"
         "  frame_size must be >= %d (GICOV interior = size - %d).\n",
         2 * MAXR + 1, 2 * MAXR);
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "s:t:h")) != -1) {
    switch (c) {
    case 's': FrameSize = atoi(optarg); break;
    case 't': MgvfIters = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (FrameSize < 2 * MAXR + 1 || MgvfIters < 1) {
    printf("Error: need frame_size >= %d and iterations >= 1\n", 2 * MAXR + 1);
    exit(-1);
  }
}

// ---------------------------------------------------------------------------
// Host-side input generation
// ---------------------------------------------------------------------------
// Small LCG so the input never depends on the process-wide rand() (PoCL also
// consumes it, which would make simx/rtlsim diverge).
struct Lcg {
  uint32_t s;
  explicit Lcg(uint32_t seed) : s(seed) {}
  double next() {
    s = s * 1664525u + 1013904223u;
    return (double)(s >> 8) * (1.0 / 16777216.0);
  }
};

// Same stencil as upstream misc_math.c: one-sided at the borders, central
// difference inside. Upstream computes in double and casts to float; the
// flattened layout is column-major: grad[col * m + row].
static void compute_gradients(const std::vector<double>& img, int m, int n,
                              std::vector<float>& gx, std::vector<float>& gy) {
  gx.assign(m * n, 0.0f);
  gy.assign(m * n, 0.0f);
  for (int r = 0; r < m; ++r)
    for (int c = 0; c < n; ++c) {
      double v;
      if (c == 0)
        v = img[r * n + 1] - img[r * n + 0];
      else if (c == n - 1)
        v = img[r * n + n - 1] - img[r * n + n - 2];
      else
        v = (img[r * n + c + 1] - img[r * n + c - 1]) / 2.0;
      gx[c * m + r] = (float)v;
    }
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r) {
      double v;
      if (r == 0)
        v = img[(r + 1) * n + c] - img[r * n + c];
      else if (r == m - 1)
        v = img[r * n + c] - img[(r - 1) * n + c];
      else
        v = (img[(r + 1) * n + c] - img[(r - 1) * n + c]) / 2.0;
      gy[c * m + r] = (float)v;
    }
}

// Same as upstream compute_constants(): numpy-free sin/cos/tX/tY tables.
static void compute_ellipse_constants(std::vector<float>& sinA,
                                      std::vector<float>& cosA,
                                      std::vector<int>& tX,
                                      std::vector<int>& tY) {
  std::vector<double> theta(NPOINTS);
  sinA.resize(NPOINTS);
  cosA.resize(NPOINTS);
  for (int n = 0; n < NPOINTS; ++n) {
    theta[n] = ((double)n * 2.0 * PI_D) / (double)NPOINTS;
    sinA[n] = (float)sin(theta[n]);
    cosA[n] = (float)cos(theta[n]);
  }
  tX.assign(NCIRCLES * NPOINTS, 0);
  tY.assign(NCIRCLES * NPOINTS, 0);
  for (int k = 0; k < NCIRCLES; ++k) {
    double rad = (double)(MIN_RAD + 2 * k);
    for (int n = 0; n < NPOINTS; ++n) {
      tX[k * NPOINTS + n] = (int)(cos(theta[n]) * rad);
      tY[k * NPOINTS + n] = (int)(sin(theta[n]) * rad);
    }
  }
}

// Same as upstream structuring_element(12).
static void compute_strel(std::vector<float>& strel) {
  strel.assign(STREL_SIZE * STREL_SIZE, 0.0f);
  for (int i = 0; i < STREL_SIZE; ++i)
    for (int j = 0; j < STREL_SIZE; ++j) {
      int di = i - STREL_RADIUS, dj = j - STREL_RADIUS;
      strel[i * STREL_SIZE + j] =
          (sqrtf((float)(di * di + dj * dj)) <= STREL_RADIUS) ? 1.0f : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------
static std::vector<float> gicov_cpu(const std::vector<float>& gx,
                                    const std::vector<float>& gy,
                                    const std::vector<float>& sinA,
                                    const std::vector<float>& cosA,
                                    const std::vector<int>& tX,
                                    const std::vector<int>& tY,
                                    int grad_m, int grad_n) {
  std::vector<float> out(grad_m * grad_n, 0.0f);
  const int width = grad_m - 2 * MAXR;
  const int height = grad_n - 2 * MAXR;
  const int total = width * height;

  for (int gid = 0; gid < total; ++gid) {
    const int i = gid / width + MAXR;
    const int j = gid % width + MAXR;

    float max_gicov = 0.0f;
    for (int k = 0; k < NCIRCLES; ++k) {
      float sum = 0.0f, M2 = 0.0f, mean = 0.0f;
      for (int n = 0; n < NPOINTS; ++n) {
        int y = j + tY[k * NPOINTS + n];
        int x = i + tX[k * NPOINTS + n];
        int addr = x * grad_m + y;
        float p = gx[addr] * cosA[n] + gy[addr] * sinA[n];
        sum += p;
        float delta = p - mean;
        mean = mean + (delta / (float)(n + 1));
        M2 = M2 + (delta * (p - mean));
      }
      mean = sum / (float)NPOINTS;
      float var = M2 / (float)(NPOINTS - 1);
      float g = mean * mean / var;
      if (g > max_gicov) max_gicov = g;
    }
    out[i * grad_m + j] = max_gicov;
  }
  return out;
}

static std::vector<float> dilate_cpu(const std::vector<float>& img, int img_m,
                                     int img_n, const std::vector<float>& strel,
                                     int strel_m, int strel_n) {
  std::vector<float> out(img_m * img_n, 0.0f);
  const int el_center_i = strel_m / 2;
  const int el_center_j = strel_n / 2;

  for (int tid = 0; tid < img_m * img_n; ++tid) {
    const int i = tid % img_m;
    const int j = tid / img_m;
    if (j >= img_n) continue;

    float mx = 0.0f;
    for (int el_i = 0; el_i < strel_m; ++el_i) {
      int y = i - el_center_i + el_i;
      if (y < 0 || y >= img_m) continue;
      for (int el_j = 0; el_j < strel_n; ++el_j) {
        int x = j - el_center_j + el_j;
        if (x < 0 || x >= img_n) continue;
        if (strel[el_i * strel_n + el_j] == 0.0f) continue;
        float t = img[x * img_m + y];
        if (t > mx) mx = t;
      }
    }
    out[i * img_n + j] = mx;
  }
  return out;
}

static inline float heaviside_f(float x) {
  return (atanf(x) * (1.0f / PI_F)) + 0.5f;
}

// MGVF Jacobi iteration over the cell image I (m x n), starting from zero.
static std::vector<float> imgvf_cpu(const std::vector<float>& I, int m, int n,
                                    float vx, float vy, float e, int iters) {
  const float MU = 0.5f;
  const float LAMBDA = 8.0f * MU + 1.0f;
  const float one_over_e = 1.0f / e;

  std::vector<float> cur(m * n, 0.0f), nxt(m * n, 0.0f);
  for (int it = 0; it < iters; ++it) {
    for (int idx = 0; idx < m * n; ++idx) {
      const int i = idx / n;
      const int j = idx % n;
      const int rowU = (i == 0) ? 0 : i - 1;
      const int rowD = (i == m - 1) ? m - 1 : i + 1;
      const int colL = (j == 0) ? 0 : j - 1;
      const int colR = (j == n - 1) ? n - 1 : j + 1;

      const float old_val = cur[idx];
      const float U = cur[rowU * n + j] - old_val;
      const float D = cur[rowD * n + j] - old_val;
      const float L = cur[i * n + colL] - old_val;
      const float R = cur[i * n + colR] - old_val;
      const float UR = cur[rowU * n + colR] - old_val;
      const float DR = cur[rowD * n + colR] - old_val;
      const float UL = cur[rowU * n + colL] - old_val;
      const float DL = cur[rowD * n + colL] - old_val;

      const float UHe = heaviside_f((U * -vy) * one_over_e);
      const float DHe = heaviside_f((D * vy) * one_over_e);
      const float LHe = heaviside_f((L * -vx) * one_over_e);
      const float RHe = heaviside_f((R * vx) * one_over_e);
      const float URHe = heaviside_f((UR * (vx - vy)) * one_over_e);
      const float DRHe = heaviside_f((DR * (vx + vy)) * one_over_e);
      const float ULHe = heaviside_f((UL * (-vx - vy)) * one_over_e);
      const float DLHe = heaviside_f((DL * (-vx + vy)) * one_over_e);

      float new_val = old_val +
          (MU / LAMBDA) * (UHe * U + DHe * D + LHe * L + RHe * R +
                           URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);
      const float vI = I[idx];
      new_val -= ((1.0f / LAMBDA) * vI * (new_val - vI));
      nxt[idx] = new_val;
    }
    cur.swap(nxt);
  }
  return cur;
}

// NaN/Inf-aware comparison: both NaN or the same-sign infinity count as equal.
static bool float_mismatch(float a, float b, float atol, float rtol) {
  if (isnan(a) || isnan(b))
    return !(isnan(a) && isnan(b));
  if (isinf(a) || isinf(b))
    return !(isinf(a) && isinf(b) && a == b);
  return !(fabsf(a - b) <= atol + rtol * fabsf(a));
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  const int grad_m = FrameSize;
  const int grad_n = FrameSize;
  const int interior_w = grad_m - 2 * MAXR;
  const int interior_h = grad_n - 2 * MAXR;
  printf("Leukocyte: frame=%dx%d gicov_interior=%dx%d mgvf=%dx%d iters=%d\n",
         grad_m, grad_n, interior_h, interior_w, M_CELL, N_CELL, MgvfIters);

  // ---- synthetic frame -----------------------------------------------------
  std::vector<double> img((size_t)grad_m * grad_n);
  Lcg lcg(12345u);
  for (int r = 0; r < grad_m; ++r)
    for (int c = 0; c < grad_n; ++c) {
      double x = c, y = r;
      double bg = 0.2 + 0.05 * sin(2.0 * PI_D * x / grad_n) *
                          cos(2.0 * PI_D * y / grad_m);
      double dx = x - grad_n / 2.0, dy = y - grad_m / 2.0;
      double ell = (dx * dx) / (16.0 * 16.0) + (dy * dy) / (11.0 * 11.0);
      double val = bg + (ell <= 1.0 ? 0.7 : 0.0) + 0.01 * (lcg.next() - 0.5);
      img[r * grad_n + c] = val;
    }

  std::vector<float> gx, gy;
  compute_gradients(img, grad_m, grad_n, gx, gy);

  std::vector<float> sinA, cosA;
  std::vector<int> tX, tY;
  compute_ellipse_constants(sinA, cosA, tX, tY);

  std::vector<float> strel;
  compute_strel(strel);

  // ---- one cell image for MGVF, normalized like upstream MGVF() ------------
  std::vector<double> cell((size_t)M_CELL * N_CELL);
  for (int i = 0; i < M_CELL; ++i)
    for (int j = 0; j < N_CELL; ++j) {
      double dy = (i - 10) / 5.0;
      double dx = (j - 20) / 13.0;
      cell[i * N_CELL + j] = exp(-(dx * dx + dy * dy)) + 0.02 * (lcg.next() - 0.5);
    }
  double Imax = cell[0], Imin = cell[0];
  for (double v : cell) {
    if (v > Imax) Imax = v;
    else if (v < Imin) Imin = v;
  }
  const double eps = pow(2.0, -52.0);
  const double scale = 1.0 / (Imax - Imin + eps);
  std::vector<float> cell_f(cell.size());
  for (size_t i = 0; i < cell.size(); ++i)
    cell_f[i] = (float)((cell[i] - Imin) * scale);

  const float vx = 0.3f, vy = 0.7f;
  const float mgvf_e = 1e-10f;

  printf("Create context\n");
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  printf("Allocate device buffers\n");
  d_gx = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * gx.size(), NULL, &_err));
  d_gy = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(float) * gy.size(), NULL, &_err));
  d_sin = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * sinA.size(), NULL, &_err));
  d_cos = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * cosA.size(), NULL, &_err));
  d_tX = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(int) * tX.size(), NULL, &_err));
  d_tY = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(int) * tY.size(), NULL, &_err));
  d_gicov = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(float) * grad_m * grad_n, NULL, &_err));
  d_strel = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(float) * strel.size(), NULL, &_err));
  d_dilated = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       sizeof(float) * grad_m * grad_n, NULL, &_err));
  d_cell = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                    sizeof(float) * cell_f.size(), NULL, &_err));
  d_imgvf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     sizeof(float) * cell_f.size(), NULL, &_err));
  int host_offset = 0, host_m = M_CELL, host_n = N_CELL;
  d_offsets = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       sizeof(int), NULL, &_err));
  d_m = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(int), NULL, &_err));
  d_n = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(int), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  char build_opts[64];
  snprintf(build_opts, sizeof(build_opts), "-DLOCAL_SIZE=%d", LOCAL_SIZE);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));

  gicov_kernel = CL_CHECK2(clCreateKernel(program, "gicov_kernel", &_err));
  dilate_kernel = CL_CHECK2(clCreateKernel(program, "dilate_kernel", &_err));
  imgvf_kernel = CL_CHECK2(clCreateKernel(program, "imgvf_kernel", &_err));
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_gx, CL_TRUE, 0,
                                sizeof(float) * gx.size(), gx.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_gy, CL_TRUE, 0,
                                sizeof(float) * gy.size(), gy.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_sin, CL_TRUE, 0,
                                sizeof(float) * sinA.size(), sinA.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_cos, CL_TRUE, 0,
                                sizeof(float) * cosA.size(), cosA.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_tX, CL_TRUE, 0,
                                sizeof(int) * tX.size(), tX.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_tY, CL_TRUE, 0,
                                sizeof(int) * tY.size(), tY.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_strel, CL_TRUE, 0,
                                sizeof(float) * strel.size(), strel.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_cell, CL_TRUE, 0,
                                sizeof(float) * cell_f.size(), cell_f.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_offsets, CL_TRUE, 0,
                                sizeof(int), &host_offset, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_m, CL_TRUE, 0,
                                sizeof(int), &host_m, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_n, CL_TRUE, 0,
                                sizeof(int), &host_n, 0, NULL, NULL));

  size_t local_work_size = LOCAL_SIZE;

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();

  // ---- GICOV ---------------------------------------------------------------
  std::vector<float> zero_gicov((size_t)grad_m * grad_n, 0.0f);
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_gicov, CL_TRUE, 0,
                                sizeof(float) * zero_gicov.size(), zero_gicov.data(),
                                0, NULL, NULL));
  CL_CHECK(clSetKernelArg(gicov_kernel, 0, sizeof(int), &grad_m));
  CL_CHECK(clSetKernelArg(gicov_kernel, 1, sizeof(cl_mem), &d_gx));
  CL_CHECK(clSetKernelArg(gicov_kernel, 2, sizeof(cl_mem), &d_gy));
  CL_CHECK(clSetKernelArg(gicov_kernel, 3, sizeof(cl_mem), &d_sin));
  CL_CHECK(clSetKernelArg(gicov_kernel, 4, sizeof(cl_mem), &d_cos));
  CL_CHECK(clSetKernelArg(gicov_kernel, 5, sizeof(cl_mem), &d_tX));
  CL_CHECK(clSetKernelArg(gicov_kernel, 6, sizeof(cl_mem), &d_tY));
  CL_CHECK(clSetKernelArg(gicov_kernel, 7, sizeof(cl_mem), &d_gicov));
  CL_CHECK(clSetKernelArg(gicov_kernel, 8, sizeof(int), &interior_w));
  CL_CHECK(clSetKernelArg(gicov_kernel, 9, sizeof(int), &interior_h));

  size_t gicov_total = (size_t)interior_w * interior_h;
  size_t gicov_global =
      ((gicov_total + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, gicov_kernel, 1, NULL,
                                  &gicov_global, &local_work_size, 0, NULL, NULL));

  // ---- dilate --------------------------------------------------------------
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_dilated, CL_TRUE, 0,
                                sizeof(float) * zero_gicov.size(), zero_gicov.data(),
                                0, NULL, NULL));
  CL_CHECK(clSetKernelArg(dilate_kernel, 0, sizeof(int), &grad_m));
  CL_CHECK(clSetKernelArg(dilate_kernel, 1, sizeof(int), &grad_n));
  int strel_m = STREL_SIZE, strel_n = STREL_SIZE;
  CL_CHECK(clSetKernelArg(dilate_kernel, 2, sizeof(int), &strel_m));
  CL_CHECK(clSetKernelArg(dilate_kernel, 3, sizeof(int), &strel_n));
  CL_CHECK(clSetKernelArg(dilate_kernel, 4, sizeof(cl_mem), &d_strel));
  CL_CHECK(clSetKernelArg(dilate_kernel, 5, sizeof(cl_mem), &d_gicov));
  CL_CHECK(clSetKernelArg(dilate_kernel, 6, sizeof(cl_mem), &d_dilated));

  size_t dilate_total = (size_t)grad_m * grad_n;
  size_t dilate_global =
      ((dilate_total + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, dilate_kernel, 1, NULL,
                                  &dilate_global, &local_work_size, 0, NULL, NULL));

  // ---- IMGVF ---------------------------------------------------------------
  std::vector<float> zero_imgvf((size_t)M_CELL * N_CELL, 0.0f);
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_imgvf, CL_TRUE, 0,
                                sizeof(float) * zero_imgvf.size(), zero_imgvf.data(),
                                0, NULL, NULL));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 0, sizeof(cl_mem), &d_imgvf));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 1, sizeof(cl_mem), &d_cell));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 2, sizeof(cl_mem), &d_offsets));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 3, sizeof(cl_mem), &d_m));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 4, sizeof(cl_mem), &d_n));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 5, sizeof(float), &vx));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 6, sizeof(float), &vy));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 7, sizeof(float), &mgvf_e));
  CL_CHECK(clSetKernelArg(imgvf_kernel, 8, sizeof(int), &MgvfIters));

  size_t imgvf_global = LOCAL_SIZE;  // one work-group (one cell)
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, imgvf_kernel, 1, NULL,
                                  &imgvf_global, &local_work_size, 0, NULL, NULL));

  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  std::vector<float> h_gicov(grad_m * grad_n), h_dilated(grad_m * grad_n),
      h_imgvf(M_CELL * N_CELL);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_gicov, CL_TRUE, 0,
                               sizeof(float) * h_gicov.size(), h_gicov.data(),
                               0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_dilated, CL_TRUE, 0,
                               sizeof(float) * h_dilated.size(), h_dilated.data(),
                               0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_imgvf, CL_TRUE, 0,
                               sizeof(float) * h_imgvf.size(), h_imgvf.data(),
                               0, NULL, NULL));

  printf("Verify result\n");
  std::vector<float> ref_gicov =
      gicov_cpu(gx, gy, sinA, cosA, tX, tY, grad_m, grad_n);
  std::vector<float> ref_dilated =
      dilate_cpu(ref_gicov, grad_m, grad_n, strel, STREL_SIZE, STREL_SIZE);
  std::vector<float> ref_imgvf =
      imgvf_cpu(cell_f, M_CELL, N_CELL, vx, vy, mgvf_e, MgvfIters);

  const float atol = 1e-3f;
  const float rtol = 1e-3f;
  int errors = 0;

  struct Check {
    const char* name;
    const std::vector<float>& ref;
    const std::vector<float>& got;
  } checks[] = {
      {"GICOV", ref_gicov, h_gicov},
      {"dilated", ref_dilated, h_dilated},
      {"IMGVF", ref_imgvf, h_imgvf},
  };

  for (const auto& ch : checks) {
    int bad = 0;
    for (size_t i = 0; i < ch.ref.size(); ++i) {
      float a = ch.ref[i], b = ch.got[i];
      if (float_mismatch(a, b, atol, rtol)) {
        if (errors < 100)
          printf("*** error: %s[%zu] expected=%f, actual=%f\n", ch.name, i, a, b);
        ++errors;
        ++bad;
      }
    }
    printf("  %-8s : %s (%zu values)\n", ch.name,
           bad == 0 ? "OK" : "MISMATCH", ch.ref.size());
  }

  cleanup();
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}