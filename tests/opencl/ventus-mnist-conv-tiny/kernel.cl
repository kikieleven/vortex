// conv — generic 2D convolution with stride and optional ReLU.
//
//   output[oc,y,x] = bias[oc] + sum_{ic,ky,kx} input[ic, y*sh+ky, x*sw+kx] * weight[oc,ic,ky,kx]
//
// Ported from
//   third_benchmark/Ventus-OpenCL-Testcase/_get_case/MNIST_conv_tiny/conv.cl
// (byte-identical to _get_case/MNIST_conv_small/conv.cl). The indexing, the
// bias-then-accumulate order and the `if (do_relu && sum < 0) sum = 0;`
// epilogue are unchanged. One line was added:
//
//   if (out_y >= out_h || out_x >= out_w) return;
//
// Upstream has no output guard, which is fine only because it launches with an
// exact global size. Vortex caps a work-group at 16 work-items, so the host
// rounds dims 1 and 2 up to the 4x4 local size; the extra work-items would
// otherwise write past the end of `output` (out_index = oc*OH*OW + y*OW + x).
// Dimension 0 is never rounded because the local size in dim 0 is 1, so
// `bias[out_c]` stays in range. With output extents that are multiples of 4 the
// guard never fires.
//
// `max`/comparison use plain float arithmetic; `sum < 0` is the upstream form.
__kernel void conv(
    __global const float* input,   // [in_channels, in_h, in_w]
    __global const float* weight,  // [out_channels, in_channels, kernel_h, kernel_w]
    __global const float* bias,    // [out_channels]
    __global float* output,        // [out_channels, out_h, out_w]
    const int in_channels,
    const int in_h,
    const int in_w,
    const int kernel_h,
    const int kernel_w,
    const int out_h,
    const int out_w,
    const int do_relu,
    const int stride_h,
    const int stride_w
)
{
    int out_c = get_global_id(0);
    int out_y = get_global_id(1);
    int out_x = get_global_id(2);

    if (out_y >= out_h || out_x >= out_w) return;

    float sum = bias[out_c];

    for (int in_c = 0; in_c < in_channels; in_c++) {
        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                int in_y = out_y * stride_h + ky;
                int in_x = out_x * stride_w + kx;

                if (in_y >= in_h || in_x >= in_w)
                    continue; // boundary check (no overflow)

                int in_index = in_c * (in_h * in_w) + in_y * in_w + in_x;
                int weight_index = out_c * (in_channels * kernel_h * kernel_w)
                                 + in_c * (kernel_h * kernel_w)
                                 + ky * kernel_w + kx;
                sum += input[in_index] * weight[weight_index];
            }
        }
    }

    if (do_relu && sum < 0)
        sum = 0;

    int out_index = out_c * (out_h * out_w) + out_y * out_w + out_x;
    output[out_index] = sum;
}