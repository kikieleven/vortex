#ifndef COMMON_H
#define COMMON_H

/* Shared by the host and kernel.cl (upstream src/opencl/macros.h). */
#define PIx2 6.2831853071795864769252867665590058f

#define NC 4

/* Upstream uses 256 for both. Vortex caps a work-group at 16 work-items:
 * ComputePhiMag_GPU runs one work-item per element (16), and ComputeQ_GPU
 * runs NC=4 elements per work-item (64 / NC = 16). The kernel indexes blocks
 * with these constants, so the per-block coverage shrinks but the per-element
 * math is unchanged. */
#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 16
#define KERNEL_Q_THREADS_PER_BLOCK 64

#define KERNEL_Q_K_ELEMS_PER_GRID 1024

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

#endif