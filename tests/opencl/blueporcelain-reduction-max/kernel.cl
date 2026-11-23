// Blue Porcelain basic/REDUCTION_MAX — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/REDUCTION_MAX/reduction_max.cu
// Same two-level maximum: per-block local-memory tree reduction, then the host
// takes the maximum of the per-block maxima.
//
// Adaptations for Vortex: work-groups are 16 wide instead of 512 (so the host
// combines ceil(n/16) partials), and out-of-range lanes load INT_MIN instead of
// leaving local memory uninitialised. The upstream `i + s < n` guard is kept.

#define GROUP_SIZE 16
#define INT_MIN_VALUE (-2147483647 - 1)

__kernel void max_reduce(__global const int *arr,
                         __global int *out,
                         int n)
{
  __local int s_data[GROUP_SIZE];

  uint tid = get_local_id(0);
  uint group_id = get_group_id(0);
  uint i = group_id * GROUP_SIZE + tid;

  s_data[tid] = (i < (uint)n) ? arr[i] : INT_MIN_VALUE;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
    if (tid < (uint)s && i + s < (uint)n) {
      if (s_data[tid] < s_data[tid + s])
        s_data[tid] = s_data[tid + s];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0)
    out[group_id] = s_data[0];
}