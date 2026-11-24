#include "common.h"

__kernel void sigmoid(__global const TYPE *input,
                      __global TYPE *output,
                      int n)
{
  int gid = get_global_id(0);
  if (gid >= n)
    return;

  TYPE x = input[gid];
  output[gid] = (TYPE)1 / ((TYPE)1 + exp(-x));
}
