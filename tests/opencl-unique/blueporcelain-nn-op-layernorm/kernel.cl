#include "common.h"

/* One work-item per row: normalise the row's own statistics, then affine. */
__kernel void layernorm(__global const TYPE *input,
                        __global const TYPE *weight,
                        __global const TYPE *bias,
                        __global TYPE *output,
                        int cols,
                        TYPE eps)
{
  int row = get_global_id(0);
  int base = row * cols;

  TYPE sum = (TYPE)0;
  for (int j = 0; j < cols; ++j)
    sum += input[base + j];
  TYPE mean = sum / (TYPE)cols;

  TYPE var = (TYPE)0;
  for (int j = 0; j < cols; ++j) {
    TYPE d = input[base + j] - mean;
    var += d * d;
  }
  var /= (TYPE)cols;

  TYPE inv = (TYPE)1 / sqrt(var + eps);
  for (int j = 0; j < cols; ++j) {
    TYPE xhat = (input[base + j] - mean) * inv;
    output[base + j] = xhat * weight[j] + bias[j];
  }
}
