// pool2d — max / average pooling over a [C,IH,IW] tensor.
//
// Ported from third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/Pool2D/pool2d.cl.
// The indexing, the window scan, the max path, the avg path (including the
// `count_include_pad` behaviour), the bounds guard and the empty-window
// fallback are unchanged. One mechanical edit:
//
//   * `FLT_MAX` was replaced with an equal literal: the PoCL-Vortex OpenCL C
//     header set does not ship <float.h> (`'float.h' file not found`) and
//     FLT_MAX is not predefined. `-3.402823466e+38f` is the nearest float to
//     FLT_MAX, i.e. exactly FLT_MAX, so the max-pooling accumulator starts at
//     the same value as upstream.
//
// `fmax` is the float overload selected by the float arguments.
#define POOL_NEG_FLT_MAX (-3.402823466e+38f)

__kernel void pool2d(
    __global const float* x,    // [C,IH,IW]
    __global float*       y,    // [C,OH,OW]
    const int C,
    const int IH,
    const int IW,
    const int OH,
    const int OW,
    const int KH,
    const int KW,
    const int SH,
    const int SW,
    const int PH,
    const int PW,
    const int mode,                // 0=max, 1=avg
    const int count_include_pad    // 0/1
){
    const int c = get_global_id(0);
    const int oy= get_global_id(1);
    const int ox= get_global_id(2);
    if (c >= C || oy >= OH || ox >= OW) return;

    const int inHW  = IH * IW;
    float acc = (mode==0)? POOL_NEG_FLT_MAX : 0.0f;
    int count = 0;

    for (int ky = 0; ky < KH; ++ky){
        const int iy = oy * SH - PH + ky;
        for (int kx = 0; kx < KW; ++kx){
            const int ix = ox * SW - PW + kx;

            const int valid = (iy >= 0 && iy < IH && ix >= 0 && ix < IW);
            if (mode == 0){
                // max
                if (valid){
                    const int idx = c * inHW + iy * IW + ix;
                    float v = x[idx];
                    acc = fmax(acc, v);
                }
            }else{
                // avg
                float v = 0.0f;
                if (valid){
                    const int idx = c * inHW + iy * IW + ix;
                    v = x[idx];
                }
                if (count_include_pad){
                    acc += v;
                    ++count;         // every one of KH*KW counts
                }else{
                    if (valid){       // only the in-bounds window counts
                        acc += v;
                        ++count;
                    }
                }
            }
        }
    }

    if (mode == 1){
        if (count > 0) acc /= (float)count;
        else           acc = 0.0f;
    }

    y[c * (OH*OW) + oy * OW + ox] = acc;
}