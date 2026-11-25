// Leukocyte kernels, ported from Rodinia OpenCL
// (third_benchmark/gpu-rodinia/opencl/leukocyte/OpenCL).
//
// Scope (same reduction as the existing heartwall port): only the three GPU
// kernels are covered — GICOV detection, dilation, and the motion gradient
// vector field (MGVF). The upstream host's AVI reader, meschach matrix library,
// active-contour snake, and cell detection/tracking drivers are out of scope.
//
// The device has no image/TEX sampling path here, so the upstream USE_IMAGE
// branches are dropped and the kernels read gradients from global buffers.
//
// Vortex mapping: work-groups of LOCAL_SIZE (16). The upstream IMGVF kernel
// pipelines its update through local buffers with a per-virtual-block
// write-back; this port uses a clean Jacobi double buffer instead (every read
// comes from the previous iteration), which is the intended MGVF math and makes
// the CPU reference deterministic. The convergence early-exit is not exercised.

#define NPOINTS 150
#define MAX_RAD 20
#define NCIRCLES 7
#define MAXR (MAX_RAD + 2)

#define STREL_RADIUS 12
#define STREL_SIZE (STREL_RADIUS * 2 + 1)

// Upper bounds for the local MGVF buffers (one work-group per cell).
#define M_MAX 21
#define N_MAX 41

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 16
#endif

#define PI 3.14159f
#define ONE_OVER_PI (1.0f / PI)
#define MU 0.5f
#define LAMBDA (8.0f * MU + 1.0f)

static float heaviside(float x) {
  return (atan(x) * ONE_OVER_PI) + 0.5f;
}

// GICOV score at every interior pixel of the gradient frame.
__kernel void gicov_kernel(int grad_m,
                           __global const float* grad_x,
                           __global const float* grad_y,
                           __constant const float* c_sin_angle,
                           __constant const float* c_cos_angle,
                           __constant const int* c_tX,
                           __constant const int* c_tY,
                           __global float* gicov,
                           int width,
                           int height) {
  const int gid = get_global_id(0);
  if (gid >= width * height)
    return;

  const int i = gid / width + MAXR;
  const int j = gid % width + MAXR;

  float max_gicov = 0.0f;
  for (int k = 0; k < NCIRCLES; ++k) {
    float sum = 0.0f, M2 = 0.0f, mean = 0.0f;
    for (int n = 0; n < NPOINTS; ++n) {
      const int y = j + c_tY[k * NPOINTS + n];
      const int x = i + c_tX[k * NPOINTS + n];
      const int addr = x * grad_m + y;
      const float p = grad_x[addr] * c_cos_angle[n] +
                      grad_y[addr] * c_sin_angle[n];

      sum += p;
      const float delta = p - mean;
      mean = mean + (delta / (float)(n + 1));
      M2 = M2 + (delta * (p - mean));
    }
    mean = sum / (float)(NPOINTS);
    const float var = M2 / (float)(NPOINTS - 1);
    const float g = mean * mean / var;
    if (g > max_gicov)
      max_gicov = g;
  }

  gicov[i * grad_m + j] = max_gicov;
}

// Dilation of the GICOV matrix by the circular structuring element.
__kernel void dilate_kernel(int img_m,
                            int img_n,
                            int strel_m,
                            int strel_n,
                            __constant const float* c_strel,
                            __global const float* img,
                            __global float* dilated) {
  const int el_center_i = strel_m / 2;
  const int el_center_j = strel_n / 2;

  const int thread_id = get_global_id(0);
  const int i = thread_id % img_m;
  const int j = thread_id / img_m;
  if (j >= img_n)
    return;

  float max = 0.0f;
  for (int el_i = 0; el_i < strel_m; ++el_i) {
    const int y = i - el_center_i + el_i;
    if (y < 0 || y >= img_m)
      continue;
    for (int el_j = 0; el_j < strel_n; ++el_j) {
      const int x = j - el_center_j + el_j;
      if (x < 0 || x >= img_n)
        continue;
      if (c_strel[el_i * strel_n + el_j] == 0.0f)
        continue;
      const float temp = img[x * img_m + y];
      if (temp > max)
        max = temp;
    }
  }

  dilated[i * img_n + j] = max;
}

// One work-group per cell. Jacobi MGVF iteration over the cell image.
__kernel void imgvf_kernel(__global float* IMGVF_array,
                           __global const float* I_array,
                           __constant const int* I_offsets,
                           __constant const int* m_array,
                           __constant const int* n_array,
                           float vx,
                           float vy,
                           float e,
                           int max_iterations) {
  const int cell = get_group_id(0);
  const int tid = get_local_id(0);

  __global float* IMGVF = IMGVF_array + I_offsets[cell];
  __global const float* I = I_array + I_offsets[cell];
  const int m = m_array[cell];
  const int n = n_array[cell];
  const int total = m * n;

  __local float cur[M_MAX * N_MAX];
  __local float nxt[M_MAX * N_MAX];

  for (int idx = tid; idx < total; idx += LOCAL_SIZE)
    cur[idx] = IMGVF[idx];
  barrier(CLK_LOCAL_MEM_FENCE);

  const float one_over_e = 1.0f / e;

  for (int it = 0; it < max_iterations; ++it) {
    for (int idx = tid; idx < total; idx += LOCAL_SIZE) {
      const int i = idx / n;
      const int j = idx % n;

      const int rowU = (i == 0) ? 0 : i - 1;
      const int rowD = (i == m - 1) ? m - 1 : i + 1;
      const int colL = (j == 0) ? 0 : j - 1;
      const int colR = (j == n - 1) ? n - 1 : j + 1;

      const float old_val = cur[idx];
      const float U  = cur[rowU * n + j] - old_val;
      const float D  = cur[rowD * n + j] - old_val;
      const float L  = cur[i * n + colL] - old_val;
      const float R  = cur[i * n + colR] - old_val;
      const float UR = cur[rowU * n + colR] - old_val;
      const float DR = cur[rowD * n + colR] - old_val;
      const float UL = cur[rowU * n + colL] - old_val;
      const float DL = cur[rowD * n + colL] - old_val;

      const float UHe  = heaviside((U * -vy) * one_over_e);
      const float DHe  = heaviside((D * vy) * one_over_e);
      const float LHe  = heaviside((L * -vx) * one_over_e);
      const float RHe  = heaviside((R * vx) * one_over_e);
      const float URHe = heaviside((UR * (vx - vy)) * one_over_e);
      const float DRHe = heaviside((DR * (vx + vy)) * one_over_e);
      const float ULHe = heaviside((UL * (-vx - vy)) * one_over_e);
      const float DLHe = heaviside((DL * (-vx + vy)) * one_over_e);

      float new_val = old_val +
          (MU / LAMBDA) * (UHe * U + DHe * D + LHe * L + RHe * R +
                           URHe * UR + DRHe * DR + ULHe * UL + DLHe * DL);

      const float vI = I[idx];
      new_val -= ((1.0f / LAMBDA) * vI * (new_val - vI));
      nxt[idx] = new_val;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int idx = tid; idx < total; idx += LOCAL_SIZE)
      cur[idx] = nxt[idx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int idx = tid; idx < total; idx += LOCAL_SIZE)
    IMGVF[idx] = cur[idx];
}