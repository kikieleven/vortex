// Blue Porcelain basic/RADIX_SORT — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/RADIX_SORT/radix_sort.cu
// Same bit-by-bit stable partition over 32 bits: for each bit, flag the values,
// build the down/up destination indices, and scatter.
//
// Adaptations for Vortex:
//   - `generateFlag` is one work-item per element. Upstream launches a block per
//     256 elements but each CUDA thread loops over the whole array, which is not
//     the parallel form the algorithm describes; the intended one-work-item-per-
//     element mapping is used here.
//   - the permutation itself is a kernel (`scatter`), where upstream performs it
//     on the host after computing the indices there. Destination indices are
//     unique, so no atomics are needed.
//   - `iDown`/`iUp` remain host-side sequential passes, as upstream.

__kernel void generate_flag(__global const int *arr,
                            __global int *flag,
                            int n,
                            int bit)
{
  int i = get_global_id(0);
  if (i < n)
    flag[i] = (arr[i] >> bit) & 1;
}

__kernel void scatter(__global const int *src,
                      __global int *dst,
                      __global const int *index,
                      int n)
{
  int i = get_global_id(0);
  if (i < n)
    dst[index[i]] = src[i];
}