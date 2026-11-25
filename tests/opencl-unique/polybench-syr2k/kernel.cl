// syr2k (PolyBench-ACC linear-algebra/kernels/syr2k) — Vortex port.
//
//   C := alpha * A * B^T + alpha * B * A^T + beta * C
//
// C, A and B are n x n. PolyBench declares NI (C dimension) and NJ (A/B row
// length) separately, but its OpenCL kernel indexes C with stride nj and A with
// stride ni — that only lines up when NI == NJ. Using one size keeps host and
// kernel consistent and matches PolyBench when NI == NJ.
//
// PolyBench's OpenCL kernel fuses the two updates into one `+=` expression while
// its CPU reference performs two separate `+=` statements. The two are
// algebraically identical but round differently, so this port uses the CPU
// reference's two-statement order on both sides to keep them bit-comparable.
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

__kernel void syr2k_kernel(__global const float *a,
                           __global const float *b,
                           __global float *c,
                           float alpha, float beta,
                           int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    float acc = beta * c[i * n + j];
    for (int k = 0; k < n; ++k) {
      acc += alpha * a[i * n + k] * b[j * n + k];
      acc += alpha * b[i * n + k] * a[j * n + k];
    }
    c[i * n + j] = acc;
  }
}
