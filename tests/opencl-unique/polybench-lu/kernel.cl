// lu (PolyBench-ACC linear-algebra/solvers/lu) — Vortex port.
//
//   for k in 0..n-1:
//     kernel1: A[k][j] /= A[k][k]                       for j > k
//     kernel2: A[i][j] -= A[i][k] * A[k][j]             for i,j > k
//
// PolyBench launches the two kernels once per k; this port does the same on the
// in-order queue, which preserves the dependency between the row scaling and the
// trailing update.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void lu_kernel1(__global float *A, int k, int n)
{
  int j = get_global_id(0) + (k + 1);

  if (j < n) {
    A[k * n + j] = A[k * n + j] / A[k * n + k];
  }
}

__kernel void lu_kernel2(__global float *A, int k, int n)
{
  int j = get_global_id(0) + (k + 1);
  int i = get_global_id(1) + (k + 1);

  if ((i < n) && (j < n)) {
    A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
  }
}
