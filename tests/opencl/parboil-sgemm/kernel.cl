// Parboil sgemm (src/opencl_base/kernel.cl) — dense single-precision
// matrix-matrix product, C = alpha*A*B^T + beta*C.
//
// Ported faithfully: the upstream OpenCL version is the *naive* per-output
// kernel (one work-item computes one C element by looping over k); it is not
// the register-tiled CUDA variant. The math, column-major indexing and
// alpha/beta epilogue are unchanged.
//
// Adaptation for Vortex:
//   - upstream launches a TILE_SZ x TILE_SZ (16x16 = 256 work-items) group;
//     Vortex caps a work-group at 16, so the host launches 4x4 groups instead.
//     The kernel itself only reads global ids, so the only change is the
//     bounds guard below (global size is rounded up to a multiple of 4; the
//     guard skips the extra work-items).
//   - float literals carry the f suffix (the device has no fp64).

__kernel void mysgemmNT( __global const float *A, int lda, __global const float *B, int ldb, __global float* C, int ldc, int k, float alpha, float beta )
{
    float c = 0.0f;
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m < k && n < k) {
        for (int i = 0; i < k; ++i) {
            float a = A[m + i * lda];
            float b = B[n + i * ldb];
            c += a * b;
        }
        C[m+n*ldc] = C[m+n*ldc] * beta + alpha * c;
    }
}