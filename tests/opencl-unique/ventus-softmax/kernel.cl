// softmax — numerically stable softmax, two layouts.
//
// Ported from third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/Softmax/softmax.cl.
// The body, the three passes (max, sum of exp, normalize) and the accumulation
// order are unchanged; `exp(...)` is the upstream spelling, kept as is.
//
// Note on the builtin: the PoCL-Vortex OpenCL C headers declare `exp` but not
// `expf`, so writing `expf` fails to build (`use of undeclared identifier`).
// In OpenCL C `exp` is overloaded on the argument type, so with a `float`
// argument it resolves to the float version and stays in the working type --
// nothing is promoted to double (the device has no cl_khr_fp64). The host
// reference uses `expf` from math.h, which is the same float function.

/* 1) Row softmax over the last dimension of an [M, N] row-major matrix. */
__kernel void softmax_rows(
    __global const float* x,   // [M*N]
    __global float*       y,   // [M*N]
    const int M,
    const int N
){
    const int r = get_global_id(0);   // row index
    if (r >= M) return;

    const int base = r * N;

    // 1) max
    float m = x[base];
    for (int j=1; j<N; ++j){
        float v = x[base + j];
        m = v > m ? v : m;
    }

    // 2) sum exp(x - m), staged into y
    float sum = 0.0f;
    for (int j=0; j<N; ++j){
        float e = exp(x[base + j] - m);
        y[base + j] = e;
        sum += e;
    }

    // 3) normalize
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int j=0; j<N; ++j){
        y[base + j] *= inv_sum;
    }
}

/* 2) Softmax over the channel dimension of an NCHW (N=1) tensor: every (y,x)
   position is independent. Index: idx = c*(H*W) + y*W + x. */
__kernel void softmax_chw(
    __global const float* x,   // [C,H,W]
    __global float*       y,   // [C,H,W]
    const int C,
    const int H,
    const int W
){
    const int yy = get_global_id(0);
    const int xx = get_global_id(1);
    if (yy >= H || xx >= W) return;

    const int HW = H * W;
    const int col = yy * W + xx;

    // 1) max over c
    float m = x[col];
    for (int c=1; c<C; ++c){
        float v = x[c*HW + col];
        m = v > m ? v : m;
    }

    // 2) sum exp(x - m), staged into y
    float sum = 0.0f;
    for (int c=0; c<C; ++c){
        float e = exp(x[c*HW + col] - m);
        y[c*HW + col] = e;
        sum += e;
    }

    // 3) normalize
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (int c=0; c<C; ++c){
        y[c*HW + col] *= inv_sum;
    }
}