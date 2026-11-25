// Blue Porcelain basic/DOT_PRODUCT — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/DOT_PRODUCT/dot_product.cu
// Same two-level reduction: each work-group accumulates a grid-strided slice of
// a[i]*b[i] in local memory, tree-reduces it, and writes one partial sum; the
// host adds the partial sums.
//
// Adaptations for Vortex:
//   - work-groups are 16 wide instead of 256, and the grid is
//     min(32, ceil(n/16)) groups (upstream: min(32, ceil(N/256))).
//   - the grid-stride loop is written with get_num_groups() so it stays correct
//     for any n.

#define GROUP_SIZE 16

__kernel void dot_product(__global const float *a,
                  __global const float *b,
                  __global float *c,
                  int n)
{
  __local float cache[GROUP_SIZE];

  int local_id = get_local_id(0);
  int group_id = get_group_id(0);
  int group_size = get_local_size(0);
  int num_groups = get_num_groups(0);

  float temp = 0.0f;
  for (int i = group_id * group_size + local_id; i < n; i += group_size * num_groups)
    temp += a[i] * b[i];

  cache[local_id] = temp;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (local_id < stride)
      cache[local_id] += cache[local_id + stride];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_id == 0)
    c[group_id] = cache[0];
}