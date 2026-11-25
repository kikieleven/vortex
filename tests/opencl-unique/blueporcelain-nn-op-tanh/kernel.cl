#include "common.h"
#undef tanh

__kernel void tanh(__global const TYPE *input,
                      __global TYPE *output,
                      int n)
{
  int gid = get_global_id(0);
  if (gid >= n)
    return;

  TYPE x = input[gid];
  output[gid] = (TYPE)1 - (TYPE)2 / (exp((TYPE)2 * x) + (TYPE)1);
}
