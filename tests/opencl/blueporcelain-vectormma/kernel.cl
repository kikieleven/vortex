// Blue Porcelain basic/VECTORMMA — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/VECTORMMA/mma_constmem.cu
// The arithmetic is unchanged: c[i] = a[i] * b[i] + d[i].
//
// Adaptation: CUDA's `__constant__ int d[N] = {1, ..., 10}` (the remaining
// entries are zero-initialised) has no portable OpenCL equivalent here, so `d`
// is passed as a `__constant` kernel argument and the host uploads the same
// 512-entry table. This still exercises the constant address space. Work-groups
// are 16 wide with an `idx < n` guard.

__kernel void device_add(__global const int *a,
                         __global const int *b,
                         __global int *c,
                         __constant const int *d,
                         int n)
{
  int index = get_global_id(0);
  if (index < n)
    c[index] = a[index] * b[index] + d[index];
}