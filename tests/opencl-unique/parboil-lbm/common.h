#ifndef COMMON_H
#define COMMON_H

/* Layout of Parboil lbm (src/opencl_base/layout_config.h + lbm_macros.h),
 * reduced to a small lattice for regression use. The stride/padding formulas
 * and the GATHER macro mapping are unchanged from upstream. */
#define SIZE_X 8
#define SIZE_Y 8
#define SIZE_Z 8

#define PADDING_X 0
#define PADDING_Y 0
#define PADDING_Z 4

#define PADDED_X (SIZE_X + PADDING_X)
#define PADDED_Y (SIZE_Y + PADDING_Y)
#define PADDED_Z (SIZE_Z + PADDING_Z)

#define TOTAL_CELLS (SIZE_X*SIZE_Y*SIZE_Z)
#define TOTAL_PADDED_CELLS (PADDED_X*PADDED_Y*PADDED_Z)

/* Distribution entries followed by the flags word (layout_config.h). */
enum { C = 0,
       N, S, E, W, T, B,
       NE, NW, SE, SW,
       NT, NB, ST, SB,
       ET, EB, WT, WB,
       FLAGS, N_CELL_ENTRIES };

#define CALC_INDEX(x,y,z,e) ((e) + N_CELL_ENTRIES*((x)+(y)*PADDED_X+(z)*PADDED_X*PADDED_Y))
#define MARGIN (CALC_INDEX(0, 0, 2, 0) - CALC_INDEX(0,0,0,0))

#define OBSTACLE    (1 << 0)
#define ACCEL       (1 << 1)
#define IN_OUT_FLOW (1 << 2)

#define OMEGA 1.95f
#define DFL1 (1.0f/ 3.0f)
#define DFL2 (1.0f/18.0f)
#define DFL3 (1.0f/36.0f)

/* lbm_macros.h: SWEEP_VAR declares the current cell, LOCAL/NEIGHBOR_* read
 * cells relative to it. */
#define SWEEP_X  __temp_x__
#define SWEEP_Y  __temp_y__
#define SWEEP_Z  __temp_z__
#define SWEEP_VAR int __temp_x__, __temp_y__, __temp_z__;

#define GRID_ENTRY(g,x,y,z,e)          ((g)[CALC_INDEX( x,  y,  z, e)])
#define GRID_ENTRY_SWEEP(g,dx,dy,dz,e) ((g)[CALC_INDEX((dx)+SWEEP_X, (dy)+SWEEP_Y, (dz)+SWEEP_Z, e)])

#define LOCAL(g,e)       (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_C(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_N(g,e)  (GRID_ENTRY_SWEEP( g,  0, +1,  0, e ))
#define NEIGHBOR_S(g,e)  (GRID_ENTRY_SWEEP( g,  0, -1,  0, e ))
#define NEIGHBOR_E(g,e)  (GRID_ENTRY_SWEEP( g, +1,  0,  0, e ))
#define NEIGHBOR_W(g,e)  (GRID_ENTRY_SWEEP( g, -1,  0,  0, e ))
#define NEIGHBOR_T(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, +1, e ))
#define NEIGHBOR_B(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, -1, e ))
#define NEIGHBOR_NE(g,e) (GRID_ENTRY_SWEEP( g, +1, +1,  0, e ))
#define NEIGHBOR_NW(g,e) (GRID_ENTRY_SWEEP( g, -1, +1,  0, e ))
#define NEIGHBOR_SE(g,e) (GRID_ENTRY_SWEEP( g, +1, -1,  0, e ))
#define NEIGHBOR_SW(g,e) (GRID_ENTRY_SWEEP( g, -1, -1,  0, e ))
#define NEIGHBOR_NT(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, +1, e ))
#define NEIGHBOR_NB(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, -1, e ))
#define NEIGHBOR_ST(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, +1, e ))
#define NEIGHBOR_SB(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, -1, e ))
#define NEIGHBOR_ET(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, +1, e ))
#define NEIGHBOR_EB(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, -1, e ))
#define NEIGHBOR_WT(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, +1, e ))
#define NEIGHBOR_WB(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, -1, e ))

/* GATHER mode (layout_config.h:45-49). */
#define SRC_C(g)  (NEIGHBOR_C ( g, C  ))
#define SRC_N(g)  (NEIGHBOR_S ( g, N  ))
#define SRC_S(g)  (NEIGHBOR_N ( g, S  ))
#define SRC_E(g)  (NEIGHBOR_W ( g, E  ))
#define SRC_W(g)  (NEIGHBOR_E ( g, W  ))
#define SRC_T(g)  (NEIGHBOR_B ( g, T  ))
#define SRC_B(g)  (NEIGHBOR_T ( g, B  ))
#define SRC_NE(g) (NEIGHBOR_SW( g, NE ))
#define SRC_NW(g) (NEIGHBOR_SE( g, NW ))
#define SRC_SE(g) (NEIGHBOR_NW( g, SE ))
#define SRC_SW(g) (NEIGHBOR_NE( g, SW ))
#define SRC_NT(g) (NEIGHBOR_SB( g, NT ))
#define SRC_NB(g) (NEIGHBOR_ST( g, NB ))
#define SRC_ST(g) (NEIGHBOR_NB( g, ST ))
#define SRC_SB(g) (NEIGHBOR_NT( g, SB ))
#define SRC_ET(g) (NEIGHBOR_WB( g, ET ))
#define SRC_EB(g) (NEIGHBOR_WT( g, EB ))
#define SRC_WT(g) (NEIGHBOR_EB( g, WT ))
#define SRC_WB(g) (NEIGHBOR_ET( g, WB ))

#define DST_C(g)  (LOCAL( g, C  ))
#define DST_N(g)  (LOCAL( g, N  ))
#define DST_S(g)  (LOCAL( g, S  ))
#define DST_E(g)  (LOCAL( g, E  ))
#define DST_W(g)  (LOCAL( g, W  ))
#define DST_T(g)  (LOCAL( g, T  ))
#define DST_B(g)  (LOCAL( g, B  ))
#define DST_NE(g) (LOCAL( g, NE ))
#define DST_NW(g) (LOCAL( g, NW ))
#define DST_SE(g) (LOCAL( g, SE ))
#define DST_SW(g) (LOCAL( g, SW ))
#define DST_NT(g) (LOCAL( g, NT ))
#define DST_NB(g) (LOCAL( g, NB ))
#define DST_ST(g) (LOCAL( g, ST ))
#define DST_SB(g) (LOCAL( g, SB ))
#define DST_ET(g) (LOCAL( g, ET ))
#define DST_EB(g) (LOCAL( g, EB ))
#define DST_WT(g) (LOCAL( g, WT ))
#define DST_WB(g) (LOCAL( g, WB ))

/* The flags word is stored as a float bit pattern (lbm_macros.h). OpenCL has
 * as_uint/as_float builtins; the host needs these helpers. */
#ifndef __OPENCL_VERSION__
static inline unsigned int as_uint(float v) {
  union { float f; unsigned int u; } t;
  t.f = v;
  return t.u;
}
static inline float as_float(unsigned int v) {
  union { float f; unsigned int u; } t;
  t.u = v;
  return t.f;
}
#endif

#endif