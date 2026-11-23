// BFS kernel — ported from SHOC level1/bfs (IIIT warp-centric method):
//   third_benchmark/shoc/src/opencl/level1/bfs/bfs_iiit.cl
//   Hong, Kim, Oguntebi, Olukotun, "Accelerating CUDA graph algorithms at
//   maximum warp", PPoPP 2011.
//
// Changes from the SHOC original:
//   - dropped the four cl_khr_*_atomics / cl_amd_printf pragmas. The kernel
//     uses no atomic builtins, and Vortex's OpenCL device does not advertise
//     those extensions.
//   - UINT_MAX is replaced by BFS_UNREACHED (0xffffffffu) to avoid depending
//     on which limits header the OpenCL C front end exposes.
//
// Work-item groups of W_SZ (32) partition one vertex's edge list by
// i = W_OFF, W_OFF+W_SZ, ... . This is a work partition, not a lock-step
// requirement, so it stays correct when a Vortex work-group is only 16 wide.

#define BFS_UNREACHED 0xffffffffu

__kernel void BFS_kernel_warp(
        __global unsigned int *levels,
        __global unsigned int *edgeArray,
        __global unsigned int *edgeArrayAux,
        int W_SZ,
        int CHUNK_SZ,
        unsigned int numVertices,
        int curr,
        __global int *flag)
{
    int tid = get_global_id(0);
    int W_OFF = tid % W_SZ;
    int W_ID = tid / W_SZ;
    int v1 = W_ID * CHUNK_SZ;
    int chk_sz = CHUNK_SZ + 1;

    if ((v1 + CHUNK_SZ) >= numVertices) {
        chk_sz = numVertices - v1 + 1;
        if (chk_sz < 0)
            chk_sz = 0;
    }

    // each warp processes CHUNK_SZ consecutive vertices, one vertex at a time
    for (int v = v1; v < chk_sz - 1 + v1; v++) {
        if (levels[v] == curr) {
            unsigned int num_nbr = edgeArray[v + 1] - edgeArray[v];
            unsigned int nbr_off = edgeArray[v];
            for (int i = W_OFF; i < num_nbr; i += W_SZ) {
                int nb = edgeArrayAux[i + nbr_off];
                if (levels[nb] == BFS_UNREACHED) {
                    levels[nb] = curr + 1;
                    *flag = 1;
                }
            }
        }
    }
}
