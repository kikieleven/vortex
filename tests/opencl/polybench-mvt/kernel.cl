// mvt (PolyBench-ACC linear-algebra/kernels/mvt) — Vortex port.
//
//   kernel1: x1 += A   * y1
//   kernel2: x2 += A^T * y2
//
// A is n x n. PolyBench declares a single size for both dimensions, so there is
// no stride ambiguity to resolve here.
//
// x1 and x2 are **input/output**: PolyBench's init_array fills them and the
// kernels accumulate on top. The host must therefore upload their initial
// values, and the comparison is against the accumulated result.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void mvt_kernel1(__global const float *a,
                          __global const float *y1,
                          __global float *x1,
                          int n)
{
  int i = get_global_id(0);

  if (i < n) {
    float acc = x1[i];
    for (int j = 0; j < n; ++j) {
      acc += a[i * n + j] * y1[j];
    }
    x1[i] = acc;
  }
}

__kernel void mvt_kernel2(__global const float *a,
                          __global const float *y2,
                          __global float *x2,
                          int n)
{
  int i = get_global_id(0);

  if (i < n) {
    float acc = x2[i];
    for (int j = 0; j < n; ++j) {
      acc += a[j * n + i] * y2[j];
    }
    x2[i] = acc;
  }
}
