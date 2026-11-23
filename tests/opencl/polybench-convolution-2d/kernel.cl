// 2D convolution (PolyBench-ACC stencils/convolution-2d) — Vortex port.
//
// B[i][j] is the 3x3 weighted sum of A centred at (i, j). Border pixels are
// not written (PolyBench semantics); the host zero-fills B before launch and
// checks that the border stays zero.
//
// The coefficients are PolyBench's, unchanged: 0.2, 0.5, -0.8, -0.3, 0.6,
// -0.9, 0.4, 0.7, 0.1. The arrangement matches PolyBench's kernel and its
// OpenMP CPU reference (PolyBench's OpenCL *host* CPU reference uses a
// transposed arrangement and is not followed here).
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`; the
// fractional coefficients would all truncate to zero under an integer type, so
// this port stays single-type like the native conv3 case.
//
// Literals carry the `f` suffix: Vortex's OpenCL device has no cl_khr_fp64, and
// an unsuffixed literal is a double in OpenCL C. `(float)0.2` and `0.2f` are
// the same value, so this does not change the arithmetic.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void Convolution2D_kernel(__global const float *A,
                                   __global float *B,
                                   int ni, int nj)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  float c11 = +0.2f;
  float c21 = +0.5f;
  float c31 = -0.8f;
  float c12 = -0.3f;
  float c22 = +0.6f;
  float c32 = -0.9f;
  float c13 = +0.4f;
  float c23 = +0.7f;
  float c33 = +0.1f;

  if ((i > 0) && (i < (ni - 1)) && (j > 0) && (j < (nj - 1))) {
    B[i * nj + j] =
        c11 * A[(i - 1) * nj + (j - 1)] +
        c21 * A[(i - 1) * nj + (j + 0)] +
        c31 * A[(i - 1) * nj + (j + 1)] +
        c12 * A[(i + 0) * nj + (j - 1)] +
        c22 * A[(i + 0) * nj + (j + 0)] +
        c32 * A[(i + 0) * nj + (j + 1)] +
        c13 * A[(i + 1) * nj + (j - 1)] +
        c23 * A[(i + 1) * nj + (j + 0)] +
        c33 * A[(i + 1) * nj + (j + 1)];
  }
}
