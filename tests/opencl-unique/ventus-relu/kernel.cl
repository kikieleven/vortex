// relu — elementwise ReLU over a [C,H,W] tensor: y = max(0, x).
//
// Ported verbatim from
//   third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/ReLU/relu.cl
// The indexing, the bounds guard and the exact comparison form
// `(v > 0.0f) ? v : 0.0f` are unchanged; only this header was added. Note the
// form matters: it maps -0.0 to +0.0, which the host reference reproduces.
__kernel void relu3d(
    __global const float* x,   // [C,H,W]
    __global float*       y,   // [C,H,W]
    const int C,
    const int H,
    const int W
){
    const int c = get_global_id(0);
    const int yy= get_global_id(1);
    const int xx= get_global_id(2);
    if (c >= C || yy >= H || xx >= W) return;

    const int idx = c * (H * W) + yy * W + xx;
    float v = x[idx];
    y[idx] = (v > 0.0f) ? v : 0.0f;
}