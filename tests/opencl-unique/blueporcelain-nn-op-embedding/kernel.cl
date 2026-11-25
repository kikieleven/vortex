#include "common.h"

__kernel void embedding(__global const TYPE *weight,
                        __global const int *indices,
                        __global TYPE *output,
                        int num_indices,
                        int dim)
{
  int gid = get_global_id(0);
  int total = num_indices * dim;
  if (gid >= total)
    return;

  int row = gid / dim;
  int col = gid % dim;
  int idx = indices[row];

  output[gid] = weight[idx * dim + col];
}
