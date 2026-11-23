// jacobi-1d-imper (PolyBench-ACC stencils/jacobi-1d-imper) — Vortex port.
//
//   for t in 0..tsteps-1:
//     kernel1: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])   for 1 <= i < n-1
//     kernel2: A[j] = B[j]                                 for 1 <= j < n-1
//
// The two kernels are launched per time step on the in-order queue. Only the
// interior is touched, so the boundaries of A keep their initial values and are
// compared too.
//
// PolyBench's OpenCL kernel already uses `0.33333f`, while its OpenMP reference
// uses a double literal. This port uses `0.33333f` on both sides so host and
// device compute the same thing.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void runJacobi1D_kernel1(__global const float *A,
                                  __global float *B,
                                  int n)
{
  int i = get_global_id(0);

  if ((i >= 1) && (i < (n - 1))) {
    B[i] = 0.33333f * (A[i - 1] + A[i] + A[i + 1]);
  }
}

__kernel void runJacobi1D_kernel2(__global float *A,
                                  __global const float *B,
                                  int n)
{
  int j = get_global_id(0);

  if ((j >= 1) && (j < (n - 1))) {
    A[j] = B[j];
  }
}
