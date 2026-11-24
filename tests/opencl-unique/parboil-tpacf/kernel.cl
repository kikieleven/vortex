// Parboil tpacf (src/opencl_base/kernel.cl) — two-point angular correlation
// function histogram.
//
// Ported faithfully: the set/sampling schedule, the dot-product distance, the
// binary search over the angular bins and the self-set pair ordering are
// unchanged.
//
// Adaptations for Vortex:
//   - BLOCK_SIZE is 16 instead of 256 and WARP_SIZE 16 instead of 32, so the
//     block's warp-histogram layout is self-consistent at Vortex's
//     16-work-item group cap (see common.h). Each work-item owns exactly one
//     histogram slot, so the upstream `atom_inc` is a plain increment.
//   - the clear loop uses a flat index and is followed by a local-memory
//     barrier; upstream relies on the two-thread-per-slot layout and has no
//     barrier there.
//   - the self-set pair filter (`tid+j > k` plus the `k < j+BLOCK_SIZE` bound)
//     is folded into the k-loop bound. The Vortex backend miscompiles the
//     original conjunction of four terms (the pair term is dropped), so this
//     form keeps the same counted pairs with a single bound.
//   - the coalescing tree over NUM_HISTOGRAMS slots is replaced by a direct
//     sum of the 16 columns per bin (same result, one pass).
//   - the final store loops over bins (tid < NUM_BINS no longer covers all 20
//     bins with 16 threads).

#include "common.h"

__kernel
void gen_hists(__global unsigned long* histograms, __global float* all_x_data,
	       __constant float* dev_binb, int NUM_SETS, int NUM_ELEMENTS)
{
  __global float* all_y_data = all_x_data + NUM_ELEMENTS*(NUM_SETS+1);
  __global float* all_z_data = all_y_data + NUM_ELEMENTS*(NUM_SETS+1);

  unsigned int bx = get_group_id(0);
  unsigned int tid = get_local_id(0);
  int do_self = (bx < (NUM_SETS + 1)) ? 1 : 0;

  __global float* data_x;
  __global float* data_y;
  __global float* data_z;
  __global float* random_x;
  __global float* random_y;
  __global float* random_z;

  __local unsigned int
    warp_hists[NUM_BINS][NUM_HISTOGRAMS]; // 20*16*4 = 1.25KB

  for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
    {
      if((w+tid) < (NUM_BINS*NUM_HISTOGRAMS))
	{
	  /* Flat clear: covers every local slot regardless of the array layout. */
	  ((__local unsigned int*)warp_hists)[w+tid] = 0;
	}
    }
  /* The upstream kernel starts accumulating without a barrier here. On
   * Vortex that lets the clear and the first increments race in local
   * memory, so make the clear visible before the accumulation. */
  barrier(CLK_LOCAL_MEM_FENCE);

  // Get stuff into shared memory to kick off the loop.
  if( do_self == 0)
    {
      data_x = all_x_data;
      data_y = all_y_data;
      data_z = all_z_data;

      random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
      random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
      random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
    }
  else
    {
      random_x = all_x_data + NUM_ELEMENTS * (bx);
      random_y = all_y_data + NUM_ELEMENTS * (bx);
      random_z = all_z_data + NUM_ELEMENTS * (bx);

      data_x = random_x;
      data_y = random_y;
      data_z = random_z;
    }

  // Iterate over all random points
  for(unsigned int j = 0; j < NUM_ELEMENTS; j += BLOCK_SIZE)
    {
      // load current random point values
      float random_x_s;
      float random_y_s;
      float random_z_s;

      if(tid + j < NUM_ELEMENTS)
        {
	  random_x_s = random_x[tid + j];
	  random_y_s = random_y[tid + j];
	  random_z_s = random_z[tid + j];
	}

      // Iterate over all data points. For a self set, the upstream pair filter
      // `tid+j > k` and bound `k < j+BLOCK_SIZE` are folded into k_end: only
      // partners with k < i are counted, and i must be a valid point.
      unsigned int k_end = 0;
      int i_idx = (int)(tid + j);
      if (i_idx < NUM_ELEMENTS) {
        if (do_self != 0) {
          k_end = (unsigned int)i_idx;
          if (k_end > j + BLOCK_SIZE)
            k_end = j + BLOCK_SIZE;
        } else {
          k_end = NUM_ELEMENTS;
        }
      }
      for(unsigned int k = 0; k < k_end; k++)
	{
	  // do actual calculations on the values:
	  float distance = data_x[k] * random_x_s +
	    data_y[k] * random_y_s +
	    data_z[k] * random_z_s ;

	  unsigned int bin_index;

	  // run binary search to find bin_index
	  unsigned int min = 0;
	  unsigned int max = NUM_BINS;
	  {
	    unsigned int k2;

	    while (max > min+1)
	      {
		k2 = (min + max) / 2;
		if (distance >= dev_binb[k2])
		  max = k2;
		else
		  min = k2;
	      }
	    bin_index = max - 1;
	  }

	  unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
	  if((distance < dev_binb[min]) && (distance >= dev_binb[max]))
	    {
	      warp_hists[bin_index][warpnum] += 1;
	    }
	}
    }

  // Coalesce the per-work-item histograms. With NUM_HISTOGRAMS == BLOCK_SIZE
  // the upstream Kogge-Stone tree over 16 slots is replaced by a direct sum of
  // the 16 columns for each bin (same result, one pass).
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  // Put the results back in the real histogram
  __global unsigned long* hist_base = histograms + NUM_BINS * bx;
  for (unsigned int b = tid; b < NUM_BINS; b += BLOCK_SIZE)
    {
      unsigned int sum = 0;
      for (int c = 0; c < NUM_HISTOGRAMS; c++)
	sum += warp_hists[b][c];
      hist_base[b] = sum;
    }
}