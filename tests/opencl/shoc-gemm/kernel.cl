// Single-precision GEMM — ported from SHOC level1/gemm/gemmN.cl.
//
// SHOC's gemmN.cl contains two Volkov register-tiled kernels (sgemmNT/sgemmNN)
// written for a 64-work-item group: local ids are 16x4, each thread holds 16 C
// accumulators, and the tile constants (64-row strips, 16-column tiles, 4
// k-values per step) are hard-coded in the body. Vortex caps a work-group at 16
// and the kernel cannot be used unchanged.
//
// Two re-blockings were attempted before this one:
//   - a 4x4-work-item / 16x16-tile register-tiled kernel (both a local-memory
//     staged version and a fully unrolled scalar-accumulator version);
//   - both abort in simx with "misaligned memory access" whose address is
//     composed of float bit patterns, i.e. the generated code uses a data value
//     as an address. The same launch shape (local size {4,4}) with a per-element
//     kernel is fine, so the trigger is the get_local_id()-based tiling itself,
//     not the launch configuration.
//
// This port therefore uses the straightforward form: one work-item per output
// element, the same NN product and the same alpha/beta epilogue as SHOC. It
// still exercises float FMA, the row/column address arithmetic, and the
// row-major layouts; it does not exercise register tiling.

#define TILE 16

__kernel void sgemmNN(__global const float *A, int lda,
                      __global const float *B, int ldb,
                      __global float *C, int ldc, int K,
                      float alpha, float beta)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    float acc = 0.0f;
    for (int kk = 0; kk < K; ++kk)
        acc += A[row * lda + kk] * B[kk * ldb + col];

    C[row * ldc + col] = alpha * acc + beta * C[row * ldc + col];
}
