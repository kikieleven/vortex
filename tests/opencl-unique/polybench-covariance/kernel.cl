// covariance (PolyBench-ACC datamining/covariance) — Vortex port.
//
//   mean, centre `data`, then the m x m covariance matrix.
//
// PolyBench declares M (variables) and N (samples) separately; using one size
// keeps host and kernel consistent and matches PolyBench when M == N.
//
// `data` is n samples x m variables, row-major.
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

__kernel void reduce_kernel(__global const float *mean,
                            __global float *data,
                            int m, int n)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < n) && (j < m)) {
    data[i * m + j] -= mean[j];
  }
}

__kernel void covar_kernel(__global float *symmat,
                           __global const float *data,
                           int m, int n)
{
  int j1 = get_global_id(0);

  if (j1 < m) {
    for (int j2 = j1; j2 < m; ++j2) {
      float acc = 0.0f;
      for (int i = 0; i < n; ++i) {
        acc += data[i * m + j1] * data[i * m + j2];
      }
      symmat[j1 * m + j2] = acc;
      symmat[j2 * m + j1] = acc;
    }
  }
}
