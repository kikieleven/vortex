// 3mm (PolyBench-ACC linear-algebra/kernels/3mm) — Vortex port.
//
//   E := A * B
//   F := C * D
//   G := E * F
//
// All matrices are n x n. PolyBench declares independent NI/NJ/NK/NL/NM sizes
// but its own host mixes the strides up; using one size keeps host and kernel
// consistent and matches PolyBench whenever every dimension is the same.
//
// Each kernel is independent (E and F do not depend on each other), so the
// host may launch them back to back on one in-order queue.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void mm3_kernel1(__global const float *A,
                          __global const float *B,
                          __global float *E,
                          int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
      acc += A[i * n + k] * B[k * n + j];
    }
    E[i * n + j] = acc;
  }
}

__kernel void mm3_kernel2(__global const float *C,
                          __global const float *D,
                          __global float *F,
                          int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
      acc += C[i * n + k] * D[k * n + j];
    }
    F[i * n + j] = acc;
  }
}

__kernel void mm3_kernel3(__global const float *E,
                          __global const float *F,
                          __global float *G,
                          int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
      acc += E[i * n + k] * F[k * n + j];
    }
    G[i * n + j] = acc;
  }
}
