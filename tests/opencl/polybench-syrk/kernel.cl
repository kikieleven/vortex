// syrk (PolyBench-ACC linear-algebra/kernels/syrk) — Vortex port.
//
//   C := alpha * A * A^T + beta * C
//
// C and A are n x n. PolyBench declares NI (C dimension) and NJ (A row length)
// separately, but its OpenCL kernel indexes C with stride nj and A with stride
// ni — that only lines up when NI == NJ. Using one size keeps host and kernel
// consistent and matches PolyBench when NI == NJ.
//
// alpha and beta are kernel arguments, as in PolyBench; the host supplies
// init_array's values (32412 / 2123).
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void syrk_kernel(__global const float *a,
                          __global float *c,
                          float alpha, float beta,
                          int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    float acc = beta * c[i * n + j];
    for (int k = 0; k < n; ++k) {
      acc += alpha * a[i * n + k] * a[j * n + k];
    }
    c[i * n + j] = acc;
  }
}
