// Blue Porcelain basic/PREFIX_SUM — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/PREFIX_SUM/scan.cu
// (the Blelloch exclusive-scan path used by prefix_sum.cu). The two kernels are
// the same algorithms: `gpu_prescan` does the upsweep/downsweep over a block of
// 2*GROUP_SIZE elements and stores the block total, `gpu_add_block_sums` adds
// each block's offset to its elements.
//
// Adaptations for Vortex:
//   - GROUP_SIZE is 16 and a block covers 32 elements (upstream: 512 / 1024).
//   - The upstream conflict-free padding (CONFLICT_FREE_OFFSET) is dropped: it
//     is `(n >> 5)`, i.e. zero for every index in a 32-element block. The
//     shmem_sz slack is therefore unnecessary too.
//   - The unused naive-scan and the second Blelloch variant in scan.cu are not
//     ported: prefix_sum.cu only calls sum_scan_blelloch().

#define GROUP_SIZE 16
#define MAX_ELEMS_PER_BLOCK (2 * GROUP_SIZE)

__kernel void gpu_prescan(__global uint *d_out,
                          __global const uint *d_in,
                          __global uint *d_block_sums,
                          uint len,
                          uint max_elems_per_block)
{
  __local uint s_out[MAX_ELEMS_PER_BLOCK];

  int thid = get_local_id(0);
  int lsize = get_local_size(0);
  uint group = get_group_id(0);
  int ai = thid;
  int bi = thid + lsize;
  uint cpy_idx = max_elems_per_block * group + thid;

  s_out[thid] = 0;
  s_out[thid + lsize] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  if (cpy_idx < len) {
    s_out[ai] = d_in[cpy_idx];
    if (cpy_idx + lsize < len)
      s_out[bi] = d_in[cpy_idx + lsize];
  }

  // Upsweep (reduce) step.
  int offset = 1;
  for (int d = max_elems_per_block >> 1; d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      int u_ai = offset * ((thid << 1) + 1) - 1;
      int u_bi = offset * ((thid << 1) + 2) - 1;
      s_out[u_bi] += s_out[u_ai];
    }
    offset <<= 1;
  }

  if (thid == 0) {
    d_block_sums[group] = s_out[max_elems_per_block - 1];
    s_out[max_elems_per_block - 1] = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Downsweep step.
  for (int d = 1; d < max_elems_per_block; d <<= 1) {
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      int d_ai = offset * ((thid << 1) + 1) - 1;
      int d_bi = offset * ((thid << 1) + 2) - 1;
      uint temp = s_out[d_ai];
      s_out[d_ai] = s_out[d_bi];
      s_out[d_bi] += temp;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if (cpy_idx < len) {
    d_out[cpy_idx] = s_out[ai];
    if (cpy_idx + lsize < len)
      d_out[cpy_idx + lsize] = s_out[bi];
  }
}

__kernel void gpu_add_block_sums(__global uint *d_out,
                                 __global const uint *d_in,
                                 __global const uint *d_block_sums,
                                 uint len)
{
  uint d_block_sum_val = d_block_sums[get_group_id(0)];
  uint cpy_idx = 2 * get_group_id(0) * get_local_size(0) + get_local_id(0);

  if (cpy_idx < len) {
    d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
    if (cpy_idx + get_local_size(0) < len)
      d_out[cpy_idx + get_local_size(0)] =
          d_in[cpy_idx + get_local_size(0)] + d_block_sum_val;
  }
}