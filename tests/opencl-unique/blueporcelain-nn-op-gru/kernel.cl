#include "common.h"

static inline TYPE gru_sigmoid(TYPE x) {
  return (TYPE)1 / ((TYPE)1 + exp(-x));
}

/* One work-item per hidden unit; weights are packed [3*hidden, in] / [3*hidden, hidden] as r,z,n. */
__kernel void gru_cell(__global const TYPE *x,
                       __global const TYPE *h_prev,
                       __global const TYPE *w_ih,
                       __global const TYPE *w_hh,
                       __global const TYPE *b_ih,
                       __global const TYPE *b_hh,
                       __global TYPE *h_next,
                       int in_features, int hidden)
{
  int j = get_global_id(0);
  if (j >= hidden)
    return;

  /* r and z are needed by unit j itself; n reuses r[j] of the same unit. */
  TYPE gi_r = b_ih[0 * hidden + j];
  TYPE gh_r = b_hh[0 * hidden + j];
  TYPE gi_z = b_ih[1 * hidden + j];
  TYPE gh_z = b_hh[1 * hidden + j];
  TYPE gi_n = b_ih[2 * hidden + j];
  TYPE gh_n = b_hh[2 * hidden + j];

  for (int k = 0; k < in_features; ++k) {
    TYPE xv = x[k];
    gi_r += w_ih[(0 * hidden + j) * in_features + k] * xv;
    gi_z += w_ih[(1 * hidden + j) * in_features + k] * xv;
    gi_n += w_ih[(2 * hidden + j) * in_features + k] * xv;
  }
  for (int k = 0; k < hidden; ++k) {
    TYPE hv = h_prev[k];
    gh_r += w_hh[(0 * hidden + j) * hidden + k] * hv;
    gh_z += w_hh[(1 * hidden + j) * hidden + k] * hv;
    gh_n += w_hh[(2 * hidden + j) * hidden + k] * hv;
  }

  TYPE r = gru_sigmoid(gi_r + gh_r);
  TYPE z = gru_sigmoid(gi_z + gh_z);
  TYPE n = tanh(gi_n + r * gh_n);

  h_next[j] = ((TYPE)1 - z) * n + z * h_prev[j];
}
