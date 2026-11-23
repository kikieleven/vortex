// atax (PolyBench-ACC linear-algebra/kernels/atax) — Vortex port.
//
//   tmp = A   * x      (tmp[i] = sum_j A[i][j] * x[j])
//   y   = A^T * tmp    (y[j]   = sum_i A[i][j] * tmp[i])
//
// A is n x n. PolyBench declares NX and NY separately; using one size keeps
// host and kernel consistent and matches PolyBench when NX == NY.
//
// Unlike PolyBench's kernels, which accumulate straight into tmp[i] / y[j] and
// therefore require the host to pre-zero both buffers, these write the finished
// accumulator. The sum order is unchanged, so the result is identical.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void atax_kernel1(__global const float *A,
                           __global const float *x,
                           __global float *tmp,
                           int n)
{
  int i = get_global_id(0);

  if (i < n) {
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      acc += A[i * n + j] * x[j];
    }
    tmp[i] = acc;
  }
}

__kernel void atax_kernel2(__global const float *A,
                           __global float *y,
                           __global const float *tmp,
                           int n)
{
  int j = get_global_id(0);

  if (j < n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      acc += A[i * n + j] * tmp[i];
    }
    y[j] = acc;
  }
}
