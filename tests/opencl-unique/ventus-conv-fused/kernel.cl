// conv2d_relu — Conv2D followed by ReLU (fused kernel, NCHW layout, N=1).
//
// Ported verbatim from
//   third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/conv_fused/conv_fused.cl
// The indexing, the accumulation order (ic, kh, kw), the boundary test and the
// `max(0.0f, value)` epilogue are unchanged; only this header was added.
// There is no bias term in this upstream kernel (unlike AIops/Conv/conv.cl).
//
// `max` is the float overload selected by the float arguments.
__kernel void conv2d_relu(
    __global const float* in,        // input tensor  [IC,IH,IW]
    __global const float* w,         // weights       [OC,IC,KH,KW]
    __global float* out,             // output tensor [OC,OH,OW]
    const int IC, const int IH, const int IW,
    const int OC, const int KH, const int KW,
    const int OH, const int OW,
    const int SH, const int SW,
    const int PH, const int PW
) {
    int oc = get_global_id(0);
    int oh = get_global_id(1);
    int ow = get_global_id(2);

    if (oc >= OC || oh >= OH || ow >= OW) return;

    float value = 0.0f;

    for (int ic = 0; ic < IC; ++ic) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = oh * SH - PH + kh;
                int iw = ow * SW - PW + kw;

                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                    int in_idx = (ic * IH + ih) * IW + iw;
                    int w_idx = (oc * IC + ic) * KH * KW + kh * KW + kw;
                    value += in[in_idx] * w[w_idx];
                }
            }
        }
    }

    out[(oc * OH + oh) * OW + ow] = max(0.0f, value);
}