#include "common.h"

__kernel void adaptive_avgpool2d(__global const TYPE *input,
                                 __global TYPE *output,
                                 int n, int c, int h, int w,
                                 int oh, int ow)
{
  int gid = get_global_id(0);
  int total = n * c * oh * ow;
  if (gid >= total)
    return;

  int ox = gid % ow;
  int t = gid / ow;
  int oy = t % oh;
  t /= oh;
  int ch = t % c;
  int nb = t / c;

  int y0 = (oy * h) / oh;
  int y1 = ((oy + 1) * h + oh - 1) / oh;
  int x0 = (ox * w) / ow;
  int x1 = ((ox + 1) * w + ow - 1) / ow;

  TYPE sum = (TYPE)0;
  int count = 0;
  for (int iy = y0; iy < y1; ++iy)
    for (int ix = x0; ix < x1; ++ix) {
      sum += input[((nb * c + ch) * h + iy) * w + ix];
      ++count;
    }

  output[gid] = sum / (TYPE)count;
}
