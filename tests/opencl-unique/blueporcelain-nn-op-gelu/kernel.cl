#include "common.h"

__kernel void gelu(__global const TYPE *input,
                      __global TYPE *output,
                      int n)
{
  int gid = get_global_id(0);
  if (gid >= n)
    return;

  TYPE x = input[gid];
  output[gid] = (TYPE)0.5 * x * ((TYPE)1 + vx_erf(x / (TYPE)1.4142135623730951f));
}
