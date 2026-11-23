#ifndef COMMON_H
#define COMMON_H

/* Parboil tpacf src/opencl_base/model.h and kernel.cl constants.
 *
 * BLOCK_SIZE is reduced from 256 to 16 (Vortex's work-group cap) and
 * WARP_SIZE from 32 to 16, so the kernel's per-warp histogram slots and its
 * coalescing reduction stay self-consistent:
 *   NUM_WARPS = 16/16 = 1, HISTS_PER_WARP = 16, NUM_HISTOGRAMS = 16.
 */
#define D2R (3.14159265358979323846/180.0)
#define R2D (180.0/3.14159265358979323846)
#define R2AM (60.0*180.0/3.14159265358979323846)

#define bins_per_dec 5
#define min_arcmin 1.0
#define max_arcmin 10000.0

#define NUM_BINS 20
#define BLOCK_SIZE 16

#define WARP_SIZE 16
#define HISTS_PER_WARP 16
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)
#define NUM_HISTOGRAMS (NUM_WARPS*HISTS_PER_WARP)
#define THREADS_PER_HIST (WARP_SIZE/HISTS_PER_WARP)

#endif