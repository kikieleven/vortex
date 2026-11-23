// gesummv (PolyBench-ACC linear-algebra/kernels/gesummv) — Vortex port.
//
//   tmp = A * x
//   y   = alpha * tmp + beta * (B * x)
//
// PolyBench's kernel accumulates straight into tmp[i] / y[i] and therefore
// requires the host to pre-zero both buffers. Here the two sums live in local
// accumulators; the addition order is unchanged, so the result is identical.
// tmp is still written back because it is part of PolyBench's interface.
//
// alpha and beta are kernel arguments, as in PolyBench; the host supplies
// init_array's values (43532 / 12313).
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void gesummv_kernel(__global const float *a,
                             __global const float *b,
                             __global const float *x,
                             __global float *y,
                             __global float *tmp,
                             float alpha, float beta,
                             int n)
{
  int i = get_global_id(0);

  if (i < n) {
    float tacc = 0.0f;
    float yacc = 0.0f;
    for (int j = 0; j < n; ++j) {
      tacc += a[i * n + j] * x[j];
      yacc += b[i * n + j] * x[j];
    }
    tmp[i] = tacc;
    y[i] = alpha * tacc + beta * yacc;
  }
}
