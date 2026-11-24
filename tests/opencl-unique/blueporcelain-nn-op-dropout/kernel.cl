#include "common.h"

__kernel void dropout(__global const TYPE *input,
                      __global TYPE *output,
                      TYPE p,
                      TYPE scale,
                      int n)
{
  int gid = get_global_id(0);
  if (gid >= n)
    return;

  TYPE r = vx_random_float(vx_wang_hash((unsigned int)gid));
  output[gid] = (r < p) ? (TYPE)0 : input[gid] * scale;
}
