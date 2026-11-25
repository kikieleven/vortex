// Blue Porcelain basic/MATRIX_TRANSPOSE — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/MATRIX_TRANSPOSE/matrix_transpose.cu
// Same shared-memory tile transpose: load a TILE x TILE block with coalesced
// reads, synchronise, then write the block transposed (coalesced writes).
//
// Adaptations for Vortex:
//   - TILE is 4 (4x4 = 16 work-items, the device work-group cap) instead of 16.
//   - The upstream `__syncthreads()` sits inside `if (i < m && j < n)`, which is
//     a divergent barrier on partial tiles. Here every work-item reaches the
//     barrier and the bounds check guards only the load/store.
//   - A is row-major m x n; B is row-major n x m, so B[j*m + i] = A[i*n + j].

#define TILE 4

__kernel void parallel_transpose_mem_coalescing(__global const int *A,
                                                __global int *B,
                                                int m,
                                                int n)
{
  __local int blk[TILE][TILE];

  int tx = get_local_id(0);
  int ty = get_local_id(1);
  int bx = get_group_id(0);
  int by = get_group_id(1);

  int j = bx * TILE + tx;  // column in A
  int i = by * TILE + ty;  // row in A

  bool in_range = (i < m && j < n);
  blk[ty][tx] = in_range ? A[i * n + j] : 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  if (in_range)
    B[j * m + i] = blk[ty][tx];
}