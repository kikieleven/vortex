// Blue Porcelain basic/VECTORDIV — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/VECTORDIV/vectordiv.cu
// kernel body is unchanged (c = a / b). Adaptation: 1D work-groups of 16 and an
// `i < n` guard, because the host rounds the global size up (upstream launches
// an exact multiple of 256 threads and has no guard).

__kernel void vector_div(__global const float *a,
                         __global const float *b,
                         __global float *c,
                         int n)
{
  int i = get_global_id(0);
  if (i < n)
    c[i] = a[i] / b[i];
}