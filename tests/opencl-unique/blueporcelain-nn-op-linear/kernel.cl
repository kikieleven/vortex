#include "common.h"

/* y = x * W^T + b, with W laid out as [out_features, in_features]. */
__kernel void linear(__global const TYPE *input,
                     __global const TYPE *weight,
                     __global const TYPE *bias,
                     __global TYPE *output,
                     int batch, int in_features, int out_features)
{
  int gid = get_global_id(0);
  int total = batch * out_features;
  if (gid >= total)
    return;

  int row = gid / out_features;
  int col = gid % out_features;

  TYPE acc = bias[col];
  for (int k = 0; k < in_features; ++k)
    acc += input[row * in_features + k] * weight[col * in_features + k];

  output[gid] = acc;
}
