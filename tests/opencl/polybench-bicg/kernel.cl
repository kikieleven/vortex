// BiCG (PolyBench-ACC linear-algebra/kernels/bicg) — Vortex port.
//
//   kernel1: q = A   * p      (q[i] = sum_j A[i][j] * p[j])
//   kernel2: s = A^T * r      (s[j] = sum_i A[i][j] * r[i])
//
// A is n x n. PolyBench declares NX and NY separately; using one size keeps
// host and kernel consistent and matches PolyBench when NX == NY.
//
// PolyBench's kernels write `q[i] = 0.0` then accumulate in place, and rely on
// the host to zero `s` first. These write the finished accumulator instead; the
// sum order is unchanged, so the result is identical.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void bicg_kernel1(__global const float *A,
                           __global const float *p,
                           __global float *q,
                           int n)
{
  int i = get_global_id(0);

  if (i < n) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      acc += A[i * n + j] * p[j];
    }
    q[i] = acc;
  }
}

__kernel void bicg_kernel2(__global const float *A,
                           __global const float *r,
                           __global float *s,
                           int n)
{
  int j = get_global_id(0);

  if (j < n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      acc += A[i * n + j] * r[i];
    }
    s[j] = acc;
  }
}
