// gemm (PolyBench-ACC linear-algebra/kernels/gemm) — Vortex port.
//
//   C := alpha * A * B + beta * C
//
// All matrices are n x n. PolyBench declares NI/NJ/NK separately but its OpenCL
// host launches over (NI, NJ) and indexes B with stride nj, so the shapes only
// agree when every dimension matches; using one size keeps host and kernel
// consistent and matches PolyBench when NI == NJ == NK.
//
// The accumulator starts at beta*C and is written back once, which is the same
// sequence of operations as PolyBench's `C *= beta; C += ...` form (the
// additions happen in the same order), so the result is bit-identical.
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

__kernel void gemm(__global const float *a,
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
      acc += alpha * a[i * n + k] * b[k * n + j];
    }
    c[i * n + j] = acc;
  }
}
