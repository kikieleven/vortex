// Parboil cutcp (src/opencl_base/kernel.cl) — cutoff pair-potential lattice.
//
// Ported faithfully: the 8x8x2 region decomposition, the shifted bin origin,
// the local atom-bin cache (8 atoms per bin, 32 bins), the neighbor-list
// batching, the cutoff test and the q*rsqrt(r2)*(1-r2/cutoff2)^2 energy
// expression are unchanged.
//
// Adaptations for Vortex:
//   - upstream launches 8x8x2 = 128-work-item blocks. Vortex caps a work-group
//     at 16, so each physical group of 16 work-items iterates over the 128
//     logical thread indices (lt = tid + 16*e). The logical mapping used by
//     the shared-memory cache (half-warp = lt>>4) and the sub-region write
//     index (lt) is preserved.
//   - the int4 neighbor list is a flat int array (3 components per entry), and
//     the float4 atom bins are a flat float array (4 floats per atom); the
//     layout change avoids vector loads that the RISC-V backend cannot select.
//     NbrListLen is a scalar instead of a pointer.
//   - the kernel takes xRegionDim/yRegionDim, which upstream reads with
//     get_num_groups(); the 1D physical launch has no such group geometry.

#include "common.h"

#define GROUP_SIZE 16
#define LOGICAL_THREADS 128

__kernel void opencl_cutoff_potential_lattice(
    int binDim_x,
    int binDim_y,
    __global float *binBaseAddr,
    int offset,
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    __global float *regionZeroAddr,
    int zRegionIndex,
    int NbrListLen,
    __global const int *NbrList,   /* {x,y,z} offsets, 3 ints per entry */
    int xRegionDim,
    int yRegionDim
    )
{
  __global float* binZeroAddr = binBaseAddr + offset;

  __local float AtomBinCache[BIN_CACHE_MAXLEN * BIN_DEPTH * 4];
  __local int myBinIndex[3];

  const int tid = get_local_id(0);
  const int lws = get_local_size(0);
  const int bid = get_group_id(0);

  /* upstream group geometry was {4*xRegionDim, yRegionDim, 1} */
  const int gx = bid % (4 * xRegionDim);
  const int gy = bid / (4 * xRegionDim);

  if (tid == 0) {
    myBinIndex[0] = (int) floor((8 * (gx >> 2) + 4) * h * BIN_INVLEN);
    myBinIndex[1] = (int) floor((8 * gy + 4) * h * BIN_INVLEN);
    myBinIndex[2] = (int) floor((8 * zRegionIndex + 4) * h * BIN_INVLEN);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  /* one accumulator per logical work-item handled by this physical one */
  float energy[LOGICAL_THREADS / GROUP_SIZE];
  for (int e = 0; e < LOGICAL_THREADS / GROUP_SIZE; e++)
    energy[e] = 0.0f;

  for (int totalbins = 0; totalbins < NbrListLen; totalbins += BIN_CACHE_MAXLEN) {
    int numbins = BIN_CACHE_MAXLEN;
    if (totalbins + BIN_CACHE_MAXLEN > NbrListLen)
      numbins = NbrListLen - totalbins;

    /* cache up to BIN_CACHE_MAXLEN bins; half-warp lt>>4 loads 4 bins */
    for (int lt = tid; lt < LOGICAL_THREADS; lt += lws) {
      int startoff = BIN_SIZE * (lt >> 4);
      for (int bincnt = 0; bincnt < 4; bincnt++) {
        int nbrid = (lt >> 4) + 8 * bincnt + totalbins;
        if (nbrid >= NbrListLen)
          break;

        int i = myBinIndex[0] + NbrList[nbrid * 3 + 0];
        int j = myBinIndex[1] + NbrList[nbrid * 3 + 1];
        int k = myBinIndex[2] + NbrList[nbrid * 3 + 2];

        __global float *p_global = binZeroAddr
          + (((k * binDim_y) + j) * binDim_x + i) * BIN_SIZE;

        int tidmask = lt & 15;
        int binIndex = startoff + bincnt * 8 * BIN_SIZE;

        AtomBinCache[binIndex + tidmask] = p_global[tidmask];
        AtomBinCache[binIndex + tidmask + 16] = p_global[tidmask + 16];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int e = 0; e < LOGICAL_THREADS / GROUP_SIZE; e++) {
      int lt = tid + e * GROUP_SIZE;
      int lx = lt & 7;
      int ly = (lt >> 3) & 7;
      int lz = lt >> 6;

      float x = (8 * (gx >> 2) + lx) * h;
      float y = (8 * gy + ly) * h;
      float z = (8 * zRegionIndex + 2 * (gx & 3) + lz) * h;

      for (int bincnt = 0; bincnt < numbins; bincnt++) {
        for (int i = 0; i < BIN_DEPTH; i++) {
          float ax = AtomBinCache[bincnt * BIN_SIZE + i * 4];
          float ay = AtomBinCache[bincnt * BIN_SIZE + i * 4 + 1];
          float az = AtomBinCache[bincnt * BIN_SIZE + i * 4 + 2];
          float aq = AtomBinCache[bincnt * BIN_SIZE + i * 4 + 3];
          if (0.0f == aq) break;  /* no more atoms in bin */
          float r2 = (ax - x) * (ax - x) + (ay - y) * (ay - y) + (az - z) * (az - z);
          if (r2 < cutoff2) {
            float s = (1.0f - r2 * inv_cutoff2);
            energy[e] += aq * rsqrt(r2) * s * s;
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  __global float *mySubRegionAddr = regionZeroAddr
    + ((zRegionIndex * yRegionDim + gy) * xRegionDim + (gx >> 2)) * REGION_SIZE
    + (gx & 3) * SUB_REGION_SIZE;

  for (int e = 0; e < LOGICAL_THREADS / GROUP_SIZE; e++) {
    int lt = tid + e * GROUP_SIZE;
    mySubRegionAddr[lt] = energy[e];
  }
}