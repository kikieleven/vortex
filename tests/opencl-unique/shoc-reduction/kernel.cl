// Reduction — ported from SHOC level1/reduction/reduction.cl, single precision.
//
// SHOC compiles this kernel for float/double via SINGLE_PRECISION /
// K_DOUBLE_PRECISION. Vortex has no fp64 support and this port is
// single-precision only, so FPTYPE is float.
//
// The reduceNoLocal fallback kernel is not ported: it only exists for devices
// whose maximum work-group size is 1 (Reduction.cpp:202-206), and Vortex
// supports 16.
//
// Algorithm unchanged: each work-group reduces `2*local_size` elements per
// stride into private/local accumulators, then performs a tree reduction in
// local memory.

__kernel void
reduce(__global const float *g_idata, __global float *g_odata,
       __local float *sdata, const unsigned int n)
{
    const unsigned int tid = get_local_id(0);
    unsigned int i = (get_group_id(0) * (get_local_size(0) * 2)) + tid;
    const unsigned int gridSize = get_local_size(0) * 2 * get_num_groups(0);
    const unsigned int blockSize = get_local_size(0);

    sdata[tid] = 0;

    // Reduce multiple elements per thread, strided by grid size
    while (i < n)
    {
        sdata[tid] += g_idata[i] + g_idata[i + blockSize];
        i += gridSize;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write result back to global memory
    if (tid == 0)
    {
        g_odata[get_group_id(0)] = sdata[0];
    }
}
