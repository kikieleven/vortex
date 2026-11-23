#include "common.h"

/* One work-item per channel: reduce over that channel's whole batch slice. */
__kernel void bn_stats(__global const TYPE *input,
                       __global TYPE *mean,
                       __global TYPE *var,
                       int n, int c, int per_channel)
{
  int ch = get_global_id(0);
  if (ch >= c)
    return;

  TYPE sum = (TYPE)0;
  int count = 0;
  for (int nb = 0; nb < n; ++nb)
    for (int i = 0; i < per_channel; ++i) {
      sum += input[(nb * c + ch) * per_channel + i];
      ++count;
    }
  TYPE mu = sum / (TYPE)count;

  TYPE acc = (TYPE)0;
  for (int nb = 0; nb < n; ++nb)
    for (int i = 0; i < per_channel; ++i) {
      TYPE d = input[(nb * c + ch) * per_channel + i] - mu;
      acc += d * d;
    }

  mean[ch] = mu;
  var[ch] = acc / (TYPE)count;
}

__kernel void bn_apply(__global const TYPE *input,
                       __global const TYPE *mean,
                       __global const TYPE *var,
                       __global const TYPE *gamma,
                       __global const TYPE *beta,
                       __global TYPE *output,
                       int c, int per_channel,
                       TYPE eps)
{
  int gid = get_global_id(0);
  int ch = (gid / per_channel) % c;

  TYPE inv = (TYPE)1 / sqrt(var[ch] + eps);
  output[gid] = (input[gid] - mean[ch]) * inv * gamma[ch] + beta[ch];
}
