#include "common.h"

__kernel void relu(__global const TYPE *input,
                   __global TYPE *output,
                   int n)
{
  int gid = get_global_id(0);
  if (gid >= n)
    return;

  TYPE x = input[gid];
  output[gid] = x > (TYPE)0 ? x : (TYPE)0;
}
