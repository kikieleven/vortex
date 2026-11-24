#include "common.h"

/* One work-item per output element; NCDHW in, KCDHW weights. */
__kernel void conv3d(__global const TYPE *input,
                     __global const TYPE *weight,
                     __global const TYPE *bias,
                     __global TYPE *output,
                     int n, int c, int d, int h, int w,
                     int k, int kd, int kh, int kw,
                     int od, int oh, int ow, int pad)
{
  int gid = get_global_id(0);
  int total = n * k * od * oh * ow;
  if (gid >= total)
    return;

  int ox = gid % ow;
  int t = gid / ow;
  int oy = t % oh;
  t /= oh;
  int oz = t % od;
  t /= od;
  int ok = t % k;
  int nb = t / k;

  TYPE acc = bias[ok];
  for (int ci = 0; ci < c; ++ci)
    for (int kz = 0; kz < kd; ++kz) {
      int iz = oz + kz - pad;
      if (iz < 0 || iz >= d)
        continue;
      for (int ky = 0; ky < kh; ++ky) {
        int iy = oy + ky - pad;
        if (iy < 0 || iy >= h)
          continue;
        for (int kx = 0; kx < kw; ++kx) {
          int ix = ox + kx - pad;
          if (ix < 0 || ix >= w)
            continue;
          TYPE xv = input[(((nb * c + ci) * d + iz) * h + iy) * w + ix];
          TYPE wv = weight[(((ok * c + ci) * kd + kz) * kh + ky) * kw + kx];
          acc += xv * wv;
        }
      }
    }

  output[gid] = acc;
}
