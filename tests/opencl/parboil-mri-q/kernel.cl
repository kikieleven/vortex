// Parboil mri-q (src/opencl/kernels.cl) — the "coarse general" ComputeQ path.
//
// Ported faithfully: ComputePhiMag_GPU and ComputeQ_GPU (COARSE_GENERAL
// branch), including the NC=4 elements per work-item and the K-tile loop.
// The compile-time block constants live in common.h; they are reduced from
// 256 to 64 (ComputeQ), i.e. 16 work-items per group (Vortex's work-group
// cap), and 16 for ComputePhiMag. The per-element expressions are unchanged.

#include "common.h"

__kernel void
ComputePhiMag_GPU(__global float* phiR, __global float* phiI, __global float* phiMag, int numK) {
  int indexK = get_global_id(0);
  if (indexK < numK) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

__kernel void
ComputeQ_GPU(int numK, int kGlobalIndex,
	     __global float* x, __global float* y, __global float* z,
	     __global float* Qr, __global float* Qi,
	     __global struct kValues* ck)
{
  float sX[NC];
  float sY[NC];
  float sZ[NC];
  float sQr[NC];
  float sQi[NC];

  #pragma unroll
  for (int tx = 0; tx < NC; tx++) {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC * get_local_id(0) + tx;

    sX[tx] = x[xIndex];
    sY[tx] = y[xIndex];
    sZ[tx] = z[xIndex];
    sQr[tx] = Qr[xIndex];
    sQi[tx] = Qi[xIndex];
  }

  // Loop over all elements of K in the tile buffer to compute a partial value
  // for X.
  int kIndex = 0;
  for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
       kIndex ++, kGlobalIndex ++) {
    float kx = ck[kIndex].Kx;
    float ky = ck[kIndex].Ky;
    float kz = ck[kIndex].Kz;
    float pm = ck[kIndex].PhiMag;

    #pragma unroll
    for (int tx = 0; tx < NC; tx++) {
      float expArg = PIx2 *
                   (kx * sX[tx] +
                    ky * sY[tx] +
                    kz * sZ[tx]);
      sQr[tx] += pm * cos(expArg);
      sQi[tx] += pm * sin(expArg);
    }
  }

  #pragma unroll
  for (int tx = 0; tx < NC; tx++) {
    int xIndex = get_group_id(0)*KERNEL_Q_THREADS_PER_BLOCK + NC * get_local_id(0) + tx;
    Qr[xIndex] = sQr[tx];
    Qi[xIndex] = sQi[tx];
  }
}