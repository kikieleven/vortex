// conv2d_bn — Conv2D followed by BatchNorm (inference), NCHW layout (N=1).
//
// Ported verbatim from
//   third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/Conv_BN/conv_bn.cl
// The indexing, the bias-then-accumulate order (ic, ky, kx), the boundary
// skips and the expression order `gamma*(sum-mean)*inv_std + beta` are
// unchanged; only this header was added.
//
// `sqrt` is the float overload selected by the float argument. The
// `bias ? bias[oc] : 0.0f` guard is kept although the host always supplies a
// bias buffer (a cl_mem argument is never NULL).
__kernel void conv2d_bn(
    __global const float* input,    // [IC,H,W]
    __global const float* weight,   // [OC,IC,KH,KW]
    __global const float* bias,     // [OC] (all-zero buffer if the layer has none)
    __global const float* gamma,    // [OC]
    __global const float* beta,     // [OC]
    __global const float* mean,     // [OC]
    __global const float* var,      // [OC]
    const float eps,
    __global float*       output,   // [OC,OH,OW]
    const int IC,
    const int IH,
    const int IW,
    const int OC,
    const int KH,
    const int KW,
    const int OH,
    const int OW,
    const int SH,
    const int SW,
    const int PH,
    const int PW
){
    const int oc = get_global_id(0);
    const int oy = get_global_id(1);
    const int ox = get_global_id(2);
    if (oc >= OC || oy >= OH || ox >= OW) return;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < IC; ++ic){
        for (int ky = 0; ky < KH; ++ky){
            const int iy = oy * SH - PH + ky;
            if (iy < 0 || iy >= IH) continue;
            for (int kx = 0; kx < KW; ++kx){
                const int ix = ox * SW - PW + kx;
                if (ix < 0 || ix >= IW) continue;

                const int in_idx = ic * (IH * IW) + iy * IW + ix;
                const int w_idx  = oc * (IC * KH * KW)
                                 + ic * (KH * KW)
                                 + ky * KW + kx;

                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    const float inv_std = 1.0f / sqrt(var[oc] + eps);
    const float y = gamma[oc] * (sum - mean[oc]) * inv_std + beta[oc];

    const int out_idx = oc * (OH * OW) + oy * OW + ox;
    output[out_idx] = y;
}