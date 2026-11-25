// Blue Porcelain basic/VECTORADD — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/VECTORADD/vectoradd.cu
// Three data types as upstream (float / int / int64), one work-item per element.
// Adaptation: 1D work-groups of 16 (Vortex device cap), global size rounded up,
// and an explicit `i < n` guard (upstream relies on the 256-thread block being
// an exact divisor of N).

__kernel void vecadd_float(__global const float *a,
                           __global const float *b,
                           __global float *c,
                           int n)
{
  int i = get_global_id(0);
  if (i < n)
    c[i] = a[i] + b[i];
}

__kernel void vecadd_int(__global const int *a,
                         __global const int *b,
                         __global int *c,
                         int n)
{
  int i = get_global_id(0);
  if (i < n)
    c[i] = a[i] + b[i];
}

__kernel void vecadd_long(__global const long *a,
                          __global const long *b,
                          __global long *c,
                          int n)
{
  int i = get_global_id(0);
  if (i < n)
    c[i] = a[i] + b[i];
}