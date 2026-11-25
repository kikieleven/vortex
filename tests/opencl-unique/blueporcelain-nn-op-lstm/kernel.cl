#include "common.h"

static inline TYPE lstm_sigmoid(TYPE x) {
  return (TYPE)1 / ((TYPE)1 + exp(-x));
}

/* One work-item per hidden unit; weights packed [4*hidden, in] / [4*hidden, hidden] as i,f,g,o. */
__kernel void lstm_cell(__global const TYPE *x,
                        __global const TYPE *h_prev,
                        __global const TYPE *c_prev,
                        __global const TYPE *w_ih,
                        __global const TYPE *w_hh,
                        __global const TYPE *b_ih,
                        __global const TYPE *b_hh,
                        __global TYPE *h_next,
                        __global TYPE *c_next,
                        int in_features, int hidden)
{
  int j = get_global_id(0);
  if (j >= hidden)
    return;

  TYPE gi_i = b_ih[0 * hidden + j], gh_i = b_hh[0 * hidden + j];
  TYPE gi_f = b_ih[1 * hidden + j], gh_f = b_hh[1 * hidden + j];
  TYPE gi_g = b_ih[2 * hidden + j], gh_g = b_hh[2 * hidden + j];
  TYPE gi_o = b_ih[3 * hidden + j], gh_o = b_hh[3 * hidden + j];

  for (int k = 0; k < in_features; ++k) {
    TYPE xv = x[k];
    gi_i += w_ih[(0 * hidden + j) * in_features + k] * xv;
    gi_f += w_ih[(1 * hidden + j) * in_features + k] * xv;
    gi_g += w_ih[(2 * hidden + j) * in_features + k] * xv;
    gi_o += w_ih[(3 * hidden + j) * in_features + k] * xv;
  }
  for (int k = 0; k < hidden; ++k) {
    TYPE hv = h_prev[k];
    gh_i += w_hh[(0 * hidden + j) * hidden + k] * hv;
    gh_f += w_hh[(1 * hidden + j) * hidden + k] * hv;
    gh_g += w_hh[(2 * hidden + j) * hidden + k] * hv;
    gh_o += w_hh[(3 * hidden + j) * hidden + k] * hv;
  }

  TYPE i_gate = lstm_sigmoid(gi_i + gh_i);
  TYPE f_gate = lstm_sigmoid(gi_f + gh_f);
  TYPE g_gate = tanh(gi_g + gh_g);
  TYPE o_gate = lstm_sigmoid(gi_o + gh_o);

  TYPE c = f_gate * c_prev[j] + i_gate * g_gate;
  c_next[j] = c;
  h_next[j] = o_gate * tanh(c);
}
