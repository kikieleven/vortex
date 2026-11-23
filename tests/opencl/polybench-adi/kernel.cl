// adi (PolyBench-ACC stencils/adi) — Vortex port.
//
// Per time step, six stages, each a separate kernel / launch:
//   1. row elimination   (i1 over all rows, i2 = 1..n-1)
//   2. row last element  X[i1][n-1] /= B[i1][n-1]
//   3. row back-substitution
//   4. column elimination (i1 = 1..n-1, launched once per i1)
//   5. column last row   X[n-1][i2] /= B[n-1][i2]
//   6. column back-substitution (i1 = 0..n-3, launched once per i1)
//
// Stages 4 and 6 take i1 as a scalar argument, as in PolyBench's kernels, so
// the host loops over them on the in-order queue. Every stage depends on the
// previous one, so the launch order is the algorithm.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void adi_kernel1(__global const float *A,
                          __global float *B,
                          __global float *X,
                          int n)
{
  int i1 = get_global_id(0);

  if (i1 < n) {
    for (int i2 = 1; i2 < n; ++i2) {
      X[i1 * n + i2] = X[i1 * n + i2]
                     - X[i1 * n + (i2 - 1)] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
      B[i1 * n + i2] = B[i1 * n + i2]
                     - A[i1 * n + i2] * A[i1 * n + i2] / B[i1 * n + (i2 - 1)];
    }
  }
}

__kernel void adi_kernel2(__global const float *A,
                          __global const float *B,
                          __global float *X,
                          int n)
{
  int i1 = get_global_id(0);

  if (i1 < n) {
    X[i1 * n + (n - 1)] = X[i1 * n + (n - 1)] / B[i1 * n + (n - 1)];
  }
}

__kernel void adi_kernel3(__global const float *A,
                          __global const float *B,
                          __global float *X,
                          int n)
{
  int i1 = get_global_id(0);

  if (i1 < n) {
    for (int i2 = 0; i2 < n - 2; ++i2) {
      X[i1 * n + (n - i2 - 2)] =
          (X[i1 * n + (n - 2 - i2)] - X[i1 * n + (n - 2 - i2 - 1)] * A[i1 * n + (n - i2 - 3)])
          / B[i1 * n + (n - 3 - i2)];
    }
  }
}

__kernel void adi_kernel4(__global const float *A,
                          __global float *B,
                          __global float *X,
                          int i1, int n)
{
  int i2 = get_global_id(0);

  if (i2 < n) {
    X[i1 * n + i2] = X[i1 * n + i2]
                   - X[(i1 - 1) * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
    B[i1 * n + i2] = B[i1 * n + i2]
                   - A[i1 * n + i2] * A[i1 * n + i2] / B[(i1 - 1) * n + i2];
  }
}

__kernel void adi_kernel5(__global const float *A,
                          __global const float *B,
                          __global float *X,
                          int n)
{
  int i2 = get_global_id(0);

  if (i2 < n) {
    X[(n - 1) * n + i2] = X[(n - 1) * n + i2] / B[(n - 1) * n + i2];
  }
}

__kernel void adi_kernel6(__global const float *A,
                          __global const float *B,
                          __global float *X,
                          int i1, int n)
{
  int i2 = get_global_id(0);

  if (i2 < n) {
    X[(n - 2 - i1) * n + i2] =
        (X[(n - 2 - i1) * n + i2] - X[(n - i1 - 3) * n + i2] * A[(n - 3 - i1) * n + i2])
        / B[(n - 2 - i1) * n + i2];
  }
}
