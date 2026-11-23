// correlation (PolyBench-ACC datamining/correlation) — Vortex port.
//
//   mean, stddev, centre+normalise `data`, then the m x m correlation matrix.
//
// PolyBench declares M (variables) and N (samples) separately; using one size
// keeps host and kernel consistent and matches PolyBench when M == N.
//
// The correlation kernel writes only j1 < m-1; PolyBench's host then sets
// symmat[m-1][m-1] = 1.0 separately, which this port does the same way (a
// one-element upload after the kernel).
//
// PolyBench's corr_kernel accumulates into symmat with `+=` and relies on the
// host zero-filling it. This port uses a local accumulator instead; the sum
// order is unchanged, so the result is identical.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void mean_kernel(__global float *mean,
                          __global const float *data,
                          float float_n,
                          int m, int n)
{
  int j = get_global_id(0);

  if (j < m) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      acc += data[i * m + j];
    }
    mean[j] = acc / float_n;
  }
}

__kernel void std_kernel(__global const float *mean,
                         __global float *std,
                         __global const float *data,
                         float float_n, float eps,
                         int m, int n)
{
  int j = get_global_id(0);

  if (j < m) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      float d = data[i * m + j] - mean[j];
      acc += d * d;
    }
    acc /= float_n;
    acc = sqrt(acc);
    if (acc <= eps) {
      acc = 1.0f;
    }
    std[j] = acc;
  }
}

__kernel void reduce_kernel(__global const float *mean,
                            __global const float *std,
                            __global float *data,
                            float float_n,
                            int m, int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < m)) {
    float d = data[i * m + j] - mean[j];
    data[i * m + j] = d / (sqrt(float_n) * std[j]);
  }
}

__kernel void corr_kernel(__global float *symmat,
                          __global const float *data,
                          int m, int n)
{
  int j1 = get_global_id(0);

  if (j1 < (m - 1)) {
    symmat[j1 * m + j1] = 1.0f;

    for (int j2 = j1 + 1; j2 < m; ++j2) {
      float acc = 0.0f;
      for (int i = 0; i < n; ++i) {
        acc += data[i * m + j1] * data[i * m + j2];
      }
      symmat[j1 * m + j2] = acc;
      symmat[j2 * m + j1] = acc;
    }
  }
}
