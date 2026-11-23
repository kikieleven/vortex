// Blue Porcelain basic/MERGE_SORT — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/MERGE_SORT/merge_sort.cu
// Same bottom-up merge: for each width (2, 4, 8, ...) every thread merges one
// width-sized range, alternating source/destination buffers.
//
// Adaptations for Vortex:
//   - work-groups are 16 wide; the host launches `n` work-items (rounded up),
//     which is enough because each width doubling at most halves the number of
//     active ranges.
//   - upstream passes dim3 thread/block descriptors through device memory and
//     recomputes a linear index from them; get_global_id(0) is exactly that
//     linear index here, so the descriptors are gone.
//   - `long` stays 64-bit on both sides (OpenCL long is 64-bit).

#define GROUP_SIZE 16

__kernel void gpu_mergesort(__global const long *source,
                            __global long *dest,
                            long size,
                            long width,
                            long slices)
{
  long idx = (long)get_global_id(0);
  long start = width * idx * slices;

  for (long slice = 0; slice < slices; ++slice) {
    if (start >= size)
      break;

    long middle = (start + (width >> 1) < size) ? (start + (width >> 1)) : size;
    long end = (start + width < size) ? (start + width) : size;

    long i = start;
    long j = middle;
    for (long k = start; k < end; ++k) {
      bool cond = i < middle && (j >= end || source[i] < source[j]);
      if (cond) {
        dest[k] = source[i];
        ++i;
      } else {
        dest[k] = source[j];
        ++j;
      }
    }

    start += width;
  }
}