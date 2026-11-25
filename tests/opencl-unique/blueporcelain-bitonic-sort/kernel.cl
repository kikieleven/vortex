// Blue Porcelain basic/BITONIC_SORT — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/BITONIC_SORT/bitonic_sort.cu
// The bitonic network is unchanged inside one block: each work-item owns one
// element, `ixj = tid ^ j` pairs are compared and swapped with the direction
// selected by `tid & k`, and each block ends up sorted ascending (k == BLOCK_SIZE
// makes every `tid & k` zero).
//
// Adaptations for Vortex:
//   - BLOCK_SIZE is 16 (Vortex work-group cap) instead of 64; the grid is
//     ceil(n/16) blocks and the total element count stays n.
//   - The upstream `extern __shared__` buffer is a static `__local` array.
//   - Elements past n (when n is not a multiple of 16) are padded with INT_MAX,
//     which keeps the ascending order, and the store is guarded by `idx < n`.

#define BLOCK_SIZE 16

__kernel void bitonic_sort(__global int *values, int n)
{
  __local int s[BLOCK_SIZE];

  uint tid = get_local_id(0);
  uint offset = get_group_id(0) * BLOCK_SIZE;
  uint idx = offset + tid;

  s[tid] = (idx < (uint)n) ? values[idx] : 0x7fffffff;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint k = 2; k <= BLOCK_SIZE; k *= 2) {
    for (uint j = k / 2; j > 0; j /= 2) {
      uint ixj = tid ^ j;
      if (ixj > tid) {
        if ((tid & k) == 0) {
          if (s[ixj] < s[tid]) {
            int tmp = s[tid];
            s[tid] = s[ixj];
            s[ixj] = tmp;
          }
        } else {
          if (s[ixj] > s[tid]) {
            int tmp = s[tid];
            s[tid] = s[ixj];
            s[ixj] = tmp;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  if (idx < (uint)n)
    values[idx] = s[tid];
}