// gramschmidt (PolyBench-ACC linear-algebra/solvers/gramschmidt) — Vortex port.
//
//   for k in 0..n-1:
//     kernel1: r[k][k] = sqrt(sum_i a[i][k]^2)          (a single work-item)
//     kernel2: q[i][k] = a[i][k] / r[k][k]
//     kernel3: for j > k: r[k][j] = q[:,k] . a[:,j];  a[:,j] -= q[:,k] * r[k][j]
//
// A, Q and R are n x n. PolyBench declares NI/NJ separately; using one size
// keeps host and kernel consistent and matches PolyBench when NI == NJ.
//
// The three kernels are launched per k on the in-order queue, so the
// dependency between the norm, the division and the update is preserved.
// PolyBench launches kernel1 with a single work-item (`if (tid == 0)`), done
// here with a 1-wide NDRange.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void gramschmidt_kernel1(__global const float *a,
                                  __global float *r,
                                  __global float *q,
                                  int k, int n)
{
  int tid = get_global_id(0);

  if (tid == 0) {
    float nrm = 0.0f;
    for (int i = 0; i < n; ++i) {
      nrm += a[i * n + k] * a[i * n + k];
    }
    r[k * n + k] = sqrt(nrm);
  }
}

__kernel void gramschmidt_kernel2(__global const float *a,
                                  __global const float *r,
                                  __global float *q,
                                  int k, int n)
{
  int i = get_global_id(0);

  if (i < n) {
    q[i * n + k] = a[i * n + k] / r[k * n + k];
  }
}

__kernel void gramschmidt_kernel3(__global float *a,
                                  __global float *r,
                                  __global const float *q,
                                  int k, int n)
{
  int j = get_global_id(0) + (k + 1);

  if (j < n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      acc += q[i * n + k] * a[i * n + j];
    }
    r[k * n + j] = acc;

    for (int i = 0; i < n; ++i) {
      a[i * n + j] -= q[i * n + k] * r[k * n + j];
    }
  }
}
