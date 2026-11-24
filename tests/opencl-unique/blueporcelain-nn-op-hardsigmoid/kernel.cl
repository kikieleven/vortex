#include "common.h"

__kernel void hardsigmoid(__global const TYPE *input,
                      __global TYPE *output,
                      int n)
{
  int gid = get_global_id(0);
  if (gid >= n)
    return;

  TYPE x = input[gid];
  output[gid] = clamp(x + (TYPE)3, (TYPE)0, (TYPE)6) / (TYPE)6;
}
