// Triad — ported from SHOC level1/triad.
//
// SHOC embeds this kernel as a 7-line string in
//   third_benchmark/shoc/src/opencl/level1/triad/Triad.cpp:190-199
// The arithmetic is unchanged:
//   C[i] = A[i] + s * B[i]
// The only addition is the `gid < n` bound so the host can round the global
// size up to a work-group multiple (Vortex caps a work-group at 16).

__kernel void Triad(__global const float *memA,
                    __global const float *memB,
                    __global float *memC,
                    const float s,
                    const int n)
{
    int gid = get_global_id(0);
    if (gid < n)
        memC[gid] = memA[gid] + s * memB[gid];
}
