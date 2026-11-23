// CSR sparse matrix-vector multiply — ported from SHOC level1/spmv/spmv.cl.
//
// Ported kernel: spmv_csr_scalar_kernel (one work-item per row). SHOC also has
// spmv_csr_vector_kernel (requires a warp-sized group and a hard-coded 128-lane
// partial-sum array) and spmv_ellpackr_kernel (requires padded/column-major
// data). Neither fits Vortex's 16-work-item work-group limit without a
// rewrite, so this port keeps the scalar kernel, which is the one SHOC treats
// as the baseline.
//
// Single precision only: SHOC selects float/double with SINGLE_PRECISION /
// K_DOUBLE_PRECISION; Vortex has no fp64 support, so FPTYPE is float and the
// USE_TEXTURE image path (spmv.cl:12-25) is not ported.

__kernel void
spmv_csr_scalar_kernel(__global const float * restrict val,
                       __global const float * restrict vec,
                       __global const int * restrict cols,
                       __global const int * restrict rowDelimiters,
                       const int dim,
                       __global float * restrict out)
{
    int myRow = get_global_id(0);

    if (myRow < dim)
    {
        float t = 0;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow + 1];
        for (int j = start; j < end; j++)
        {
            int col = cols[j];
            t += val[j] * vec[col];
        }
        out[myRow] = t;
    }
}
