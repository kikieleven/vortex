// jacobi-2d-imper (PolyBench-ACC stencils/jacobi-2d-imper) — Vortex port.
//
//   for t in 0..tsteps-1:
//     kernel1: B[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i][j+1] + A[i+1][j] + A[i-1][j])
//     kernel2: A[i][j] = B[i][j]                       for 1 <= i,j < n-1
//
// The two kernels are launched per time step on the in-order queue. Only the
// interior is touched, so the boundary of A keeps its initial values and is
// compared too.
//
// PolyBench's OpenCL kernel already uses `0.2f`, while its OpenMP reference uses
// a double literal. This port uses `0.2f` on both sides.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void runJacobi2D_kernel1(__global const float *A,
                                  __global float *B,
                                  int n)
{
  int i = get_global_id(1);
  int j = get_global_id(0);

  if ((i >= 1) && (i < (n - 1)) && (j >= 1) && (j < (n - 1))) {
    B[i * n + j] = 0.2f * (A[i * n + j]
                         + A[i * n + (j - 1)]
                         + A[i * n + (1 + j)]
                         + A[(1 + i) * n + j]
                         + A[(i - 1) * n + j]);
  }
}

__kernel void runJacobi2D_kernel2(__global float *A,
                                  __global const float *B,
                                  int n)
{
  int i = get_global_id(1);
  int j = get_global_id(0);

  if ((i >= 1) && (i < (n - 1)) && (j >= 1) && (j < (n - 1))) {
    A[i * n + j] = B[i * n + j];
  }
}
