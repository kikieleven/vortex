// gemm — C = A * B with optional transpose of A / B.
//
// Ported from third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/GEMM/gemm.cl.
// The body is verbatim except for one added line:
//
//   if (row >= M || col >= N) return;
//
// The upstream kernel has no bounds guard and is launched with an exact
// global size. Vortex caps a work-group at 16 work-items, so the host must round
// the global size up to a multiple of the work-group; the guard drops the extra
// work-items. With an exact multiple (the default sizes are) the guard never
// fires and the computation is identical to upstream.
__kernel void gemm(
    __global const float* A,   // A matrix: (M x K) or (K x M) when do_trans_a
    __global const float* B,   // B matrix: (K x N) or (N x K) when do_trans_b
    __global float* C,         // output matrix (M x N)
    const int M,
    const int K,
    const int N,
    const int do_trans_a,
    const int do_trans_b
)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        float a_val = (do_trans_a) ? A[k * M + row] : A[row * K + k];
        float b_val = (do_trans_b) ? B[col * K + k] : B[k * N + col];

        sum += a_val * b_val;
    }

    C[row * N + col] = sum;
}