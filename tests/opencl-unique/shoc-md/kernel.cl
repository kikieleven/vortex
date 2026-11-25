// Lennard-Jones force — ported from SHOC level1/md/md.cl, single precision only.
//
// SHOC compiles this kernel twice (SINGLE_PRECISION / K_DOUBLE_PRECISION,
// MD.cpp:299-330). Vortex's OpenCL device has no cl_khr_fp64, and this port
// deliberately implements only the single-precision variant, so the precision
// macros are replaced by float/float4.
//
// The only addition to the kernel body is the `idx >= inum` guard, needed
// because Vortex caps a work-group at 16 and the host rounds the global size up.

__kernel void compute_lj_force(__global float4 *force,
                               __global float4 *position,
                               const int neighCount,
                               __global int *neighList,
                               const float cutsq,
                               const float lj1,
                               const float lj2,
                               const int inum)
{
    uint idx = get_global_id(0);
    if (idx >= (uint)inum)
        return;

    float4 ipos = position[idx];
    float4 f = {0.0f, 0.0f, 0.0f, 0.0f};

    int j = 0;
    while (j < neighCount)
    {
        int jidx = neighList[j * inum + idx];

        // Uncoalesced read
        float4 jpos = position[jidx];

        // Calculate distance
        float delx = ipos.x - jpos.x;
        float dely = ipos.y - jpos.y;
        float delz = ipos.z - jpos.z;
        float r2inv = delx * delx + dely * dely + delz * delz;

        // If distance is less than cutoff, calculate force
        if (r2inv < cutsq)
        {
            r2inv = 1.0f / r2inv;
            float r6inv = r2inv * r2inv * r2inv;
            float forceC = r2inv * r6inv * (lj1 * r6inv - lj2);

            f.x += delx * forceC;
            f.y += dely * forceC;
            f.z += delz * forceC;
        }
        j++;
    }
    // store the results
    force[idx] = f;
}
