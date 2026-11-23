// Blue Porcelain basic/REDUCTION_SUM — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/REDUCTION_SUM/reduction_sum.cu
// Same two-level sum: per-block local-memory tree reduction, then the host adds
// the per-block partial sums.
//
// Adaptations for Vortex:
//   - work-groups are 16 wide instead of 512, so the number of blocks is
//     ceil(n/16) and the host sums that many partials (upstream: 4).
//   - out-of-range lanes load the additive identity 0 instead of leaving local
//     memory uninitialised; the upstream `i + s < n` guard is kept.

#define GROUP_SIZE 16

__kernel void sum(__global const float *arr,
                  __global float *out,
                  int n)
{
  __local float s_data[GROUP_SIZE];

  uint tid = get_local_id(0);
  uint group_id = get_group_id(0);
  uint i = group_id * GROUP_SIZE + tid;

  s_data[tid] = (i < (uint)n) ? arr[i] : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
    if (tid < (uint)s && i + s < (uint)n)
      s_data[tid] += s_data[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0)
    out[group_id] = s_data[0];
}