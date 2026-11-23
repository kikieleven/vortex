// 3D convolution (PolyBench-ACC stencils/convolution-3d) — Vortex port.
//
// B[i][j][k] is PolyBench's 15-term stencil over A centred at (i, j, k). Only
// interior points are written (PolyBench semantics); the host zero-fills B
// before launch and checks the whole volume, so a stray write into the border
// also fails.
//
// PolyBench launches this kernel once per i plane and passes i as a scalar
// argument, because its 2D work-group only covers (k, j). Here the plane index
// is folded into the second global dimension, so one 2D launch covers the whole
// volume. A 3D NDRange is avoided because no other test in this tree uses one.
//
// IMPORTANT: the upstream kernel is degenerate — its 15 terms read
// A[i-1][j-1][k-1] three times and never read the A[i][*][k-1] row or the
// A[i+-1][*][k] rows. PolyBench's OpenMP CPU reference has exactly the same
// shape, so this is reproduced verbatim rather than "fixed"; changing it would
// break comparability with the upstream reference.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void Convolution3D_kernel(__global const float *A,
                                   __global float *B,
                                   int ni, int nj, int nk)
{
  int k  = get_global_id(0);
  int ij = get_global_id(1);

  int i = ij / nj;
  int j = ij - i * nj;

  float c11, c12, c13, c21, c22, c23, c31, c32, c33;
  c11 = +2;  c21 = +5;  c31 = -8;
  c12 = -3;  c22 = +6;  c32 = -9;
  c13 = +4;  c23 = +7;  c33 = +10;

  if ((i > 0) && (i < ni - 1) && (j > 0) && (j < nj - 1) &&
      (k > 0) && (k < nk - 1)) {
    const int plane = nk * nj;

    B[i * plane + j * nk + k] =
          c11 * A[(i - 1) * plane + (j - 1) * nk + (k - 1)]
        + c13 * A[(i + 1) * plane + (j - 1) * nk + (k - 1)]
        + c21 * A[(i - 1) * plane + (j - 1) * nk + (k - 1)]
        + c23 * A[(i + 1) * plane + (j - 1) * nk + (k - 1)]
        + c31 * A[(i - 1) * plane + (j - 1) * nk + (k - 1)]
        + c33 * A[(i + 1) * plane + (j - 1) * nk + (k - 1)]
        + c12 * A[(i + 0) * plane + (j - 1) * nk + (k + 0)]
        + c22 * A[(i + 0) * plane + (j + 0) * nk + (k + 0)]
        + c32 * A[(i + 0) * plane + (j + 1) * nk + (k + 0)]
        + c11 * A[(i - 1) * plane + (j - 1) * nk + (k + 1)]
        + c13 * A[(i + 1) * plane + (j - 1) * nk + (k + 1)]
        + c21 * A[(i - 1) * plane + (j + 0) * nk + (k + 1)]
        + c23 * A[(i + 1) * plane + (j + 0) * nk + (k + 1)]
        + c31 * A[(i - 1) * plane + (j + 1) * nk + (k + 1)]
        + c33 * A[(i + 1) * plane + (j + 1) * nk + (k + 1)];
  }
}
