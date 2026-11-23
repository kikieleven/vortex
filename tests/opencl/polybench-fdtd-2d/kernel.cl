// fdtd-2d (PolyBench-ACC stencils/fdtd-2d) — Vortex port.
//
//   for t in 0..tmax-1:
//     kernel1: ey[0][j] = fict[t];  ey[i][j] -= 0.5*(hz[i][j]-hz[i-1][j]) for i>0
//     kernel2: ex[i][j] -= 0.5*(hz[i][j]-hz[i][j-1])              for j>0
//     kernel3: hz[i][j] -= 0.7*(ex[i][j+1]-ex[i][j]+ey[i+1][j]-ey[i][j])
//
// The three kernels chain within one time step and the time steps chain too, so
// the host submits them in order on the in-order queue.
//
// PolyBench's kernel writes the constants as unsuffixed literals (`0.5`, `0.7`),
// which are doubles in OpenCL C; Vortex has no cl_khr_fp64, so this port uses
// `0.5f` / `0.7f`. The values are identical.
//
// Working type is float, matching PolyBench's `typedef float DATA_TYPE`.

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void fdtd_kernel1(__global const float *_fict_,
                           __global const float *ex,
                           __global float *ey,
                           __global const float *hz,
                           int t, int nx, int ny)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < nx) && (j < ny)) {
    if (i == 0) {
      ey[i * ny + j] = _fict_[t];
    } else {
      ey[i * ny + j] = ey[i * ny + j] - 0.5f * (hz[i * ny + j] - hz[(i - 1) * ny + j]);
    }
  }
}

__kernel void fdtd_kernel2(__global float *ex,
                           __global const float *ey,
                           __global const float *hz,
                           int nx, int ny)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < nx) && (j < ny) && (j > 0)) {
    ex[i * ny + j] = ex[i * ny + j] - 0.5f * (hz[i * ny + j] - hz[i * ny + (j - 1)]);
  }
}

__kernel void fdtd_kernel3(__global const float *ex,
                           __global const float *ey,
                           __global float *hz,
                           int nx, int ny)
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if ((i < (nx - 1)) && (j < (ny - 1))) {
    hz[i * ny + j] = hz[i * ny + j] - 0.7f * (ex[i * ny + (j + 1)] - ex[i * ny + j]
                                            + ey[(i + 1) * ny + j] - ey[i * ny + j]);
  }
}
