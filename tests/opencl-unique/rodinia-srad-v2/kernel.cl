// SRAD v2 (tiled) kernels, ported from Rodinia's CUDA implementation
// (cuda/srad/srad_v2/srad_kernel.cu). Rodinia has no upstream OpenCL version
// of srad_v2, so this is a CUDA -> OpenCL conversion.
//
// Difference from the non-tiled v1 port (tests/opencl/srad): each work-group
// stages a BLOCK_SIZE x BLOCK_SIZE tile plus the four neighbour rows/columns
// for the tile borders into local memory and barriers between the loads and
// the compute. The arithmetic, index layout (row-major, stride = cols) and
// boundary clamping are identical to v1 and to the CUDA v2 host's serial
// reference.
//
// BLOCK_SIZE must match the work-group size passed by the host (4x4 = 16),
// and rows/cols must be multiples of BLOCK_SIZE (the upstream kernel has no
// partial-tile guards).

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

// srad_cuda_1: directional derivatives, ICOV and diffusion coefficient.
// Writes C (diffusion coefficient) and the four directional derivatives.
__kernel void srad_kernel_1(__global const float* J,
                            __global float* C,
                            __global float* E_C,
                            __global float* W_C,
                            __global float* N_C,
                            __global float* S_C,
                            int cols,
                            int rows,
                            float q0sqr) {
  __local float temp[BLOCK_SIZE * BLOCK_SIZE];
  __local float c_res[BLOCK_SIZE * BLOCK_SIZE];
  __local float north[BLOCK_SIZE * BLOCK_SIZE];
  __local float south[BLOCK_SIZE * BLOCK_SIZE];
  __local float east[BLOCK_SIZE * BLOCK_SIZE];
  __local float west[BLOCK_SIZE * BLOCK_SIZE];

  const int tx = get_local_id(0);
  const int ty = get_local_id(1);
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  const int index = row * cols + col;
  const int lid = ty * BLOCK_SIZE + tx;

  // Stage the tile and the neighbour values used at tile borders. The clamped
  // forms match the upstream host's iN/iS/jW/jE boundary handling, and avoid
  // the upstream kernel's out-of-bounds reads at the image edges (which it
  // overwrote afterwards).
  const float jc = J[index];
  temp[lid] = jc;
  north[lid] = (row > 0) ? J[index - cols] : J[col];
  south[lid] = (row < rows - 1) ? J[index + cols] : J[(rows - 1) * cols + col];
  west[lid]  = (col > 0) ? J[index - 1] : J[row * cols];
  east[lid]  = (col < cols - 1) ? J[index + 1] : J[row * cols + cols - 1];
  barrier(CLK_LOCAL_MEM_FENCE);

  const float n = ((ty == 0) ? north[lid] : temp[lid - BLOCK_SIZE]) - jc;
  const float s = ((ty == BLOCK_SIZE - 1) ? south[lid] : temp[lid + BLOCK_SIZE]) - jc;
  const float w = ((tx == 0) ? west[lid] : temp[lid - 1]) - jc;
  const float e = ((tx == BLOCK_SIZE - 1) ? east[lid] : temp[lid + 1]) - jc;

  const float g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
  const float l = (n + s + w + e) / jc;
  float num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
  float den = 1 + (0.25f * l);
  const float qsqr = num / (den * den);

  // Diffusion coefficient (equ 33), saturated to [0,1].
  den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
  float c = 1.0f / (1.0f + den);
  if (c < 0) c = 0;
  else if (c > 1) c = 1;

  c_res[lid] = c;
  barrier(CLK_LOCAL_MEM_FENCE);

  C[index] = c_res[lid];
  E_C[index] = e;
  W_C[index] = w;
  S_C[index] = s;
  N_C[index] = n;
}

// srad_cuda_2: divergence of the flux and the image update.
__kernel void srad_kernel_2(__global float* J,
                            __global const float* C,
                            __global const float* E_C,
                            __global const float* W_C,
                            __global const float* N_C,
                            __global const float* S_C,
                            int cols,
                            int rows,
                            float lambda) {
  __local float temp[BLOCK_SIZE * BLOCK_SIZE];
  __local float ctemp[BLOCK_SIZE * BLOCK_SIZE];
  __local float south_c[BLOCK_SIZE * BLOCK_SIZE];
  __local float east_c[BLOCK_SIZE * BLOCK_SIZE];

  const int tx = get_local_id(0);
  const int ty = get_local_id(1);
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  const int index = row * cols + col;
  const int lid = ty * BLOCK_SIZE + tx;

  temp[lid] = J[index];
  ctemp[lid] = C[index];
  south_c[lid] = (row < rows - 1) ? C[index + cols] : C[(rows - 1) * cols + col];
  east_c[lid]  = (col < cols - 1) ? C[index + 1] : C[row * cols + cols - 1];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Same neighbour selection as the upstream kernel: the diffusion coefficient
  // is taken from the same row/column (cn = cw = c) and from the south/east
  // neighbour otherwise.
  const float cc = ctemp[lid];
  const float cn = cc;
  const float cs = (ty == BLOCK_SIZE - 1) ? south_c[lid] : ctemp[lid + BLOCK_SIZE];
  const float cw = cc;
  const float ce = (tx == BLOCK_SIZE - 1) ? east_c[lid] : ctemp[lid + 1];

  const float d_sum = cn * N_C[index] + cs * S_C[index] +
                      cw * W_C[index] + ce * E_C[index];
  J[index] = temp[lid] + 0.25f * lambda * d_sum;
}