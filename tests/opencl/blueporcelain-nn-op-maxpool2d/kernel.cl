#include "common.h"

__kernel void maxpool2d(__global const TYPE *input,
                        __global TYPE *output,
                        int n, int c, int h, int w,
                        int k, int s, int oh, int ow)
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

  int h0 = oy * s;
  int w0 = ox * s;

  TYPE best = -INFINITY;
  for (int ky = 0; ky < k; ++ky) {
    int iy = h0 + ky;
    if (iy >= h)
      continue;
    for (int kx = 0; kx < k; ++kx) {
      int ix = w0 + kx;
      if (ix >= w)
        continue;
      TYPE v = input[((nb * c + ch) * h + iy) * w + ix];
      if (v > best)
        best = v;
    }
  }

  output[gid] = best;
}
