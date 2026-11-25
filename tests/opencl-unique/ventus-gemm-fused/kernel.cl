// gemm_relu — C = max(0, A * B) with optional transpose of A / B.
//
// Ported from third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/gemm_fused/gemm_fused.cl.
// The accumulation loop, the transpose flags and the `max(0.0f, value)` epilogue
// are unchanged. Three edits, all documented here:
//
//   * the transposed index expressions were corrected. Upstream wrote
//       a_idx = (do_trans_a ? k : m) * K + (do_trans_a ? m : k)
//       b_idx = (do_trans_b ? n : k) * N + (do_trans_b ? k : n)
//     which is only right when the relevant dimensions are equal. With A
//     stored as K x M the row stride is M, not K; with B stored as N x K the
//     row stride is K, not N. Upstream's b_idx even reads out of bounds when
//     N > K (n*N+k can reach (N-1)*N+K-1 > N*K). Corrected to
//       a_idx = (do_trans_a ? k * M + m : m * K + k)
//       b_idx = (do_trans_b ? n * K + k : k * N + n)
//     The non-transposed paths are byte-for-byte the upstream expressions.
//   * a bounds guard `if (m >= M || n >= N) return;` was added: upstream has
//     none and is launched with an exact global size, but Vortex caps a
//     work-group at 16 work-items so the host rounds the global size up. With
//     the default sizes (multiples of 4) the guard never fires.
//   * this header.
//
// `max` is the float overload selected by the float arguments.
__kernel void gemm_relu(
    __global const float* A,        // A: (M x K), or (K x M) when do_trans_a
    __global const float* B,        // B: (K x N), or (N x K) when do_trans_b
    __global float* C,              // C: (M x N)
    const int M, const int K, const int N,
    const int do_trans_a, const int do_trans_b
) {
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m >= M || n >= N) return;

    float value = 0.0f;

    for (int k = 0; k < K; ++k) {
        int a_idx = (do_trans_a ? k * M + m : m * K + k);
        int b_idx = (do_trans_b ? n * K + k : k * N + n);

        value += A[a_idx] * B[b_idx];
    }

    C[m * N + n] = max(0.0f, value);
}