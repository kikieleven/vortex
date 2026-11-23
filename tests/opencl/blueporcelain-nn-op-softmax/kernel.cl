#include "common.h"

/* One work-item per row: max, exp-sum, then normalise the row in place. */
__kernel void softmax(__global const TYPE *input,
                      __global TYPE *output,
                      int cols)
{
  int row = get_global_id(0);

  TYPE best = -INFINITY;
  for (int j = 0; j < cols; ++j) {
    TYPE v = input[row * cols + j];
    if (v > best)
      best = v;
  }

  TYPE sum = (TYPE)0;
  for (int j = 0; j < cols; ++j)
    sum += exp(input[row * cols + j] - best);

  for (int j = 0; j < cols; ++j)
    output[row * cols + j] = exp(input[row * cols + j] - best) / sum;
}
