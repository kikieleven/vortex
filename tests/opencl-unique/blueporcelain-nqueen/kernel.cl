// Blue Porcelain basic/NQUEEN — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/NQUEEN/nqueen.cu
// Same non-recursive backtracking solver: each work-item receives one start
// condition (the bitmask state after the host has placed `host_mark + 1` queens)
// and counts the completions; a local-memory tree reduction sums the block.
//
// Adaptations for Vortex:
//   - GROUP_SIZE is 16 instead of THREAD_NUM=96; the per-thread stack arrays are
//     private (`uint mask[32]` ...) rather than a shared [96][10] array indexed
//     by thread id, which is the natural translation and avoids a 96-way local
//     allocation.
//   - `mark` is the upstream kernel's argument, i.e. the number of rows the
//     kernel must still place (host_mark + kernel_mark == n).
//   - n is limited to 31 (the mask shift `1u << n` must not be 32).

#define GROUP_SIZE 16

__kernel void solve_nqueen(__global const uint *total_masks,
                           __global const uint *total_l_masks,
                           __global const uint *total_r_masks,
                           __global uint *results,
                           int n,
                           int mark,
                           int total_conditions)
{
  __local uint sum[GROUP_SIZE];

  uint mask[32];
  uint l_mask[32];
  uint r_mask[32];
  uint m[32];

  int tid = get_local_id(0);
  int idx = get_global_id(0);
  uint t_mask = (1u << n) - 1;
  int total = 0;
  int i = 0;
  uint index;

  if (idx < total_conditions) {
    mask[0] = total_masks[idx];
    l_mask[0] = total_l_masks[idx];
    r_mask[0] = total_r_masks[idx];
    m[0] = mask[0] | l_mask[0] | r_mask[0];

    while (i >= 0) {
      if ((m[i] & t_mask) == t_mask) {
        i--;
      } else {
        index = (m[i] + 1) & ~m[i];
        m[i] |= index;
        if ((index & t_mask) != 0) {
          if (i + 1 == mark) {
            total++;
            i--;
          } else {
            mask[i + 1] = mask[i] | index;
            l_mask[i + 1] = (l_mask[i] | index) << 1;
            r_mask[i + 1] = (r_mask[i] | index) >> 1;
            m[i + 1] = mask[i + 1] | l_mask[i + 1] | r_mask[i + 1];
            i++;
          }
        } else {
          i--;
        }
      }
    }

    sum[tid] = (uint)total;
  } else {
    sum[tid] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s)
      sum[tid] += sum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (tid == 0)
    results[get_group_id(0)] = sum[0];
}