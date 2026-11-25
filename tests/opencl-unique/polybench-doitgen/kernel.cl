// doitgen (PolyBench-ACC linear-algebra/kernels/doitgen) — Vortex port.
//
//   sum[r][q][p] = sum_s A[r][q][s] * C4[s][p]
//   A[r][q][p]   = sum[r][q][p]
//
// PolyBench declares NR/NQ/NP separately; using one size keeps host and kernel
// consistent and matches PolyBench when all three are equal. (PolyBench's own
// CPU copy-back loop uses NR as the bound instead of NP, which is another
// reason the separate sizes cannot be trusted.)
//
// PolyBench launches kernel1/kernel2 once per r and passes r as an argument;
// this port does the same, on the in-order queue.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void doitgen_kernel1(int nr, int nq, int np,
                              __global const float *A,
                              __global const float *C4,
                              __global float *sum,
                              int r)
{
  int p = get_global_id(0);
  int q = get_global_id(1);

  if ((p < np) && (q < nq)) {
    int idx = r * (nq * np) + q * np + p;
    float acc = 0.0f;
    for (int s = 0; s < np; ++s) {
      acc += A[r * (nq * np) + q * np + s] * C4[s * np + p];
    }
    sum[idx] = acc;
  }
}

__kernel void doitgen_kernel2(int nr, int nq, int np,
                              __global float *A,
                              __global const float *C4,
                              __global const float *sum,
                              int r)
{
  int p = get_global_id(0);
  int q = get_global_id(1);

  if ((p < np) && (q < nq)) {
    A[r * (nq * np) + q * np + p] = sum[r * (nq * np) + q * np + p];
  }
}
