// 2mm (PolyBench-ACC linear-algebra/kernels/2mm) — Vortex port.
//
//   kernel1: tmp = alpha * A * B
//   kernel2: D   = beta  * D + tmp * C
//
// All matrices are n x n. PolyBench declares independent NI/NJ/NK/NL sizes but
// its OpenCL host indexes C with stride nl while the declaration gives it row
// stride nj; those disagree unless every dimension is the same. Using one size
// keeps host and kernel consistent and matches PolyBench whenever NI==NJ==NK==NL.
//
// The accumulators live in a local variable instead of being read back from
// global memory on every iteration. The additions happen in exactly the same
// order, so the result is bit-identical to PolyBench's in-place form.
//
// alpha and beta are kernel arguments, as in PolyBench's mm2_kernel*; the host
// supplies the same values PolyBench's init_array uses (32412 / 2123). Working
// type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void mm2_kernel1(__global float *tmp,
                          __global const float *A,
                          __global const float *B,
                          int n, float alpha)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
      acc += alpha * A[i * n + k] * B[k * n + j];
    }
    tmp[i * n + j] = acc;
  }
}

__kernel void mm2_kernel2(__global const float *tmp,
                          __global const float *C,
                          __global float *D,
                          int n, float beta)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    float acc = beta * D[i * n + j];
    for (int k = 0; k < n; ++k) {
      acc += tmp[i * n + k] * C[k * n + j];
    }
    D[i * n + j] = acc;
  }
}
