// gemver (PolyBench-ACC linear-algebra/kernels/gemver) — Vortex port.
//
//   kernel1: A += u1*v1^T + u2*v2^T
//   kernel2: x += beta * A^T * y + z          (x is input/output)
//   kernel3: w += alpha * A * x               (w is input/output)
//
// PolyBench's CPU reference splits `x += beta*A^T*y` and `x += z` into two
// loops; its OpenCL kernel does the same operations in one kernel (all y terms,
// then z). This port uses that same order, so the sums match.
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

__kernel void gemver_kernel1(__global float *A,
                             __global const float *V1,
                             __global const float *V2,
                             __global const float *U1,
                             __global const float *U2,
                             int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < n)) {
    A[i * n + j] += U1[i] * V1[j] + U2[i] * V2[j];
  }
}

__kernel void gemver_kernel2(__global const float *A,
                             __global float *X,
                             __global const float *Y,
                             __global const float *Z,
                             float beta, int n)
{
  int i = get_global_id(0);

  if (i < n) {
    float acc = X[i];
    for (int j = 0; j < n; ++j) {
      acc += beta * A[j * n + i] * Y[j];
    }
    acc += Z[i];
    X[i] = acc;
  }
}

__kernel void gemver_kernel3(__global const float *A,
                             __global const float *X,
                             __global float *w,
                             float alpha, int n)
{
  int i = get_global_id(0);

  if (i < n) {
    float acc = w[i];
    for (int j = 0; j < n; ++j) {
      acc += alpha * A[i * n + j] * X[j];
    }
    w[i] = acc;
  }
}
