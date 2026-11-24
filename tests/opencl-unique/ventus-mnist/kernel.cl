// fc_layer — fully-connected layer forward pass.
//
// Ported verbatim from
//   third_benchmark/Ventus-OpenCL-Testcase/_get_case/MNIST/nn_forward.cl
// The weights are stored row-major, one row per output neuron; the dot product
// accumulation order, the bias addition and the optional ReLU are unchanged;
// only this header was added.
//
// One kernel serves both upstream layers (784->128 with ReLU, 128->10 without);
// `activation` selects the ReLU. The `idx < output_size` guard lets the host
// round the global size up to the work-group size.
__kernel void fc_layer(
    const int input_size,
    const int output_size,
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    __global float* output,
    const int activation)
{
    int idx = get_global_id(0);
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i];
        }
        sum += bias[idx];
        if (activation == 1) {
            sum = fmax(sum, 0.0f);
        }
        output[idx] = sum;
    }
}