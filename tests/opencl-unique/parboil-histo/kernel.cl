// Parboil histo (src/opencl_base/histo_{prescan,intermediates,main,final}.cl)
// — 2D histogram over packed 8-bit-per-bin shared-memory counters.
//
// Ported faithfully: the prescan range estimate, the uchar4 bin descriptor
// (block, indexhi, indexlo, offset), the packed 4-bins-per-word local
// histogram with overflow accounting, the packed 16-bit global histogram for
// out-of-range pixels, and the final combine are unchanged.
//
// Adaptations for Vortex:
//   - work-groups are 16 work-items; KB = 8 so the local sub-histogram fits
//     Vortex's 16 KiB local memory. The scanning loops stride by the actual
//     local size instead of the upstream 512 threads.
//   - the prescan reduction tree is expressed as a generic halving loop for
//     PRESCAN_THREADS = 16 (same average/stddev chain).
//   - the uchar4 descriptor and the final kernel's ushort4/uint4/uchar4
//     accesses are flattened into scalar array accesses (same bytes).
//   - the local sub-histogram is a flat array indexed by indexhi*256+indexlo.
//   - atom_add on global buffers needs the Vortex A extension.

#include "common.h"

// ---------------------------------------------------------------------------
// prescan: estimate the histogram range from a 1/8 sample of the input
// ---------------------------------------------------------------------------

__kernel void histo_prescan_kernel (__global unsigned int* input, int size,
                                    __global unsigned int* minmax)
{
  __local float Avg[PRESCAN_THREADS];
  __local float StdDev[PRESCAN_THREADS];

  int threadIdxx = get_local_id(0);
  int blockDimx = get_local_size(0);
  int blockIdxx = get_group_id(0);
  int stride = size/(get_num_groups(0));
  int addr = blockIdxx*stride+threadIdxx;
  int end = blockIdxx*stride + stride/8; // Only sample 1/8th of the input

  float avg = 0.0f;
  unsigned int count = 0;
  while (addr < end){
    avg += input[addr];
    count++;
    addr += blockDimx;
  }
  avg /= count;
  Avg[threadIdxx] = avg;

  int addr2 = blockIdxx*stride+threadIdxx;
  float stddev = 0;
  while (addr2 < end){
    stddev += (input[addr2]-avg)*(input[addr2]-avg);
    addr2 += blockDimx;
  }
  stddev /= count;
  StdDev[threadIdxx] = sqrt(stddev);

  for (int s = PRESCAN_THREADS/2; s > 1; s >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (threadIdxx < s) {
      Avg[threadIdxx] += Avg[threadIdxx+s];
      StdDev[threadIdxx] += StdDev[threadIdxx+s];
    }
  }

  if (threadIdxx == 0){
    float avg2 = Avg[0]+Avg[1];
    avg2 /= PRESCAN_THREADS;
    float stddev2 = StdDev[0]+StdDev[1];
    stddev2 /= PRESCAN_THREADS;

    atomic_min(minmax, ((unsigned int)(avg2-10*stddev2))/(KB*1024));
    atomic_max(minmax+1, ((unsigned int)(avg2+10*stddev2))/(KB*1024));
  }
}

// ---------------------------------------------------------------------------
// intermediates: pixel value -> (block, indexhi, indexlo, offset) descriptor
// ---------------------------------------------------------------------------

static void calculateBin(unsigned int bin, __global unsigned char *sm_mapping)
{
  unsigned char offset  =  bin        %   4;
  unsigned char indexlo = (bin >>  2) % 256;
  unsigned char indexhi = (bin >> 10) %  KB;
  unsigned char block   =  bin / BINS_PER_BLOCK;

  offset *= 8;

  sm_mapping[0] = block;
  sm_mapping[1] = indexhi;
  sm_mapping[2] = indexlo;
  sm_mapping[3] = offset;
}

__kernel void histo_intermediates_kernel (
    __global unsigned int *input,
    unsigned int height,
    unsigned int width,
    unsigned int input_pitch,
    __global unsigned char *sm_mappings)
{
  int threadIdxx = get_local_id(0);
  int blockDimx = get_local_size(0);
  unsigned int line_base = UNROLL * (get_group_id(0));

  for (int u = 0; u < UNROLL; u++) {
    unsigned int line = line_base + u;
    if (line >= height)
      break;

    unsigned int store = line * width;

    for (unsigned int col = threadIdxx; col < input_pitch; col += blockDimx) {
      /* upstream reads a uint2 (two pixels); the RISC-V backend cannot
       * select that vector load, so read the two uints explicitly */
      unsigned int bin_x = input[line * (2 * input_pitch) + 2 * col];
      unsigned int bin_y = input[line * (2 * input_pitch) + 2 * col + 1];

      calculateBin(bin_x, &sm_mappings[(store + col) * 4]);
      bool skip = (width % 2) && (col == (input_pitch - 1));
      if (!skip)
        calculateBin(bin_y, &sm_mappings[(store + col + input_pitch) * 4]);
    }
  }
}

// ---------------------------------------------------------------------------
// main: per-range local histogram + out-of-range global histogram
// ---------------------------------------------------------------------------

static void testIncrementGlobal (
        __global unsigned int *global_histo,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        unsigned int range, unsigned int indexhi,
        unsigned int indexlo, unsigned int offset)
{
  if (range < sm_range_min || range > sm_range_max)
  {
    const unsigned int bin = range * BINS_PER_BLOCK + offset / 8 + (indexlo << 2) + (indexhi << 10);
    const unsigned int bin_div2 = bin / 2;
    const unsigned int bin_offset = (bin % 2 == 1) ? 16 : 0;

    unsigned int old_val = global_histo[bin_div2];
    unsigned short old_bin = (old_val >> bin_offset) & 0xFFFF;

    if (old_bin < 255)
    {
      atomic_add (&global_histo[bin_div2], 1 << bin_offset);
    }
  }
}

static void testIncrementLocal (
        __global unsigned int *global_overflow,
        __local unsigned int *smem,
        const unsigned int myRange,
        unsigned int range, unsigned int indexhi,
        unsigned int indexlo, unsigned int offset)
{
  if (range == myRange)
  {
    unsigned int add = (unsigned int)(1 << offset);
    unsigned int prev = atomic_add(&smem[indexhi * 256 + indexlo], add);

    unsigned int prev_bin_val = (prev >> offset) & 0x000000FF;

    if (prev_bin_val == 0x000000FF)
    {
      const unsigned int bin =
        range * BINS_PER_BLOCK +
        offset / 8 + (indexlo << 2) + (indexhi << 10);

      bool can_overflow_to_bin_plus_1 = (offset < 24) ? true : false;
      bool can_overflow_to_bin_plus_2 = (offset < 16) ? true : false;
      bool can_overflow_to_bin_plus_3 = (offset <  8) ? true : false;

      bool overflow_into_bin_plus_1 = false;
      bool overflow_into_bin_plus_2 = false;
      bool overflow_into_bin_plus_3 = false;

      unsigned int prev_bin_plus_1_val = (prev >> (offset +  8)) & 0x000000FF;
      unsigned int prev_bin_plus_2_val = (prev >> (offset + 16)) & 0x000000FF;
      unsigned int prev_bin_plus_3_val = (prev >> (offset + 24)) & 0x000000FF;

      if (can_overflow_to_bin_plus_1 &&        prev_bin_val == 0x000000FF) overflow_into_bin_plus_1 = true;
      if (can_overflow_to_bin_plus_2 && prev_bin_plus_1_val == 0x000000FF) overflow_into_bin_plus_2 = true;
      if (can_overflow_to_bin_plus_3 && prev_bin_plus_2_val == 0x000000FF) overflow_into_bin_plus_3 = true;

      unsigned int bin_plus_1_add = 0;
      unsigned int bin_plus_2_add = 0;
      unsigned int bin_plus_3_add = 0;

      if (overflow_into_bin_plus_1) bin_plus_1_add = (prev_bin_plus_1_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
      if (overflow_into_bin_plus_2) bin_plus_2_add = (prev_bin_plus_2_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
      if (overflow_into_bin_plus_3) bin_plus_3_add = (prev_bin_plus_3_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;

                                                    atomic_add (&global_overflow[bin],  256);
      if (overflow_into_bin_plus_1) atomic_add (&global_overflow[bin+1], bin_plus_1_add);
      if (overflow_into_bin_plus_2) atomic_add (&global_overflow[bin+2], bin_plus_2_add);
      if (overflow_into_bin_plus_3) atomic_add (&global_overflow[bin+3], bin_plus_3_add);
    }
  }
}

static void clearMemory (__local unsigned int *smem)
{
  for (int i = get_local_id(0), blockDimx = get_local_size(0); i < BINS_PER_BLOCK / 4; i += blockDimx)
  {
    smem[i] = 0;
  }
}

static void copyMemory (__global unsigned int *dst, __local unsigned int *src)
{
  for (int i = get_local_id(0), blockDimx = get_local_size(0); i < BINS_PER_BLOCK/4; i += blockDimx)
  {
    atomic_add(dst+i*4, (src[i] >> 0) & 0xFF );
    atomic_add(dst+i*4+1, (src[i] >> 8) & 0xFF);
    atomic_add(dst+i*4+2, (src[i] >> 16) & 0xFF);
    atomic_add(dst+i*4+3, (src[i] >> 24) & 0xFF);
  }
}

__kernel void histo_main_kernel (
        __global unsigned char *sm_mappings,
        unsigned int num_elements,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        unsigned int histo_height,
        unsigned int histo_width,
        __global unsigned int *global_subhisto,
        __global unsigned int *global_histo,
        __global unsigned int *global_overflow)
{
  __local unsigned int sub_histo[KB * 256];

  unsigned int blockDimx = get_local_size(0);
  unsigned int gridDimx = get_num_groups(0);
  unsigned int local_scan_range = sm_range_min + get_group_id(1);
  unsigned int local_scan_load = get_group_id(0) * blockDimx + get_local_id(0);

  clearMemory (sub_histo);
  barrier(CLK_LOCAL_MEM_FENCE);

  if (get_group_id(1) == 0 )
  {
    while (local_scan_load < num_elements)
    {
      unsigned int range = sm_mappings[local_scan_load * 4 + 0];
      unsigned int indexhi = sm_mappings[local_scan_load * 4 + 1];
      unsigned int indexlo = sm_mappings[local_scan_load * 4 + 2];
      unsigned int offset = sm_mappings[local_scan_load * 4 + 3];
      local_scan_load += blockDimx * gridDimx;

      testIncrementLocal (global_overflow, sub_histo, local_scan_range,
                          range, indexhi, indexlo, offset);
      testIncrementGlobal (global_histo, sm_range_min, sm_range_max,
                           range, indexhi, indexlo, offset);
    }
  }
  else
  {
    while (local_scan_load < num_elements)
    {
      unsigned int range = sm_mappings[local_scan_load * 4 + 0];
      unsigned int indexhi = sm_mappings[local_scan_load * 4 + 1];
      unsigned int indexlo = sm_mappings[local_scan_load * 4 + 2];
      unsigned int offset = sm_mappings[local_scan_load * 4 + 3];
      local_scan_load += blockDimx * gridDimx;

      testIncrementLocal (global_overflow, sub_histo, local_scan_range,
                          range, indexhi, indexlo, offset);
    }
  }

  unsigned int store_index = (local_scan_range * BINS_PER_BLOCK);

  barrier(CLK_LOCAL_MEM_FENCE);
  copyMemory (&(global_subhisto[store_index]), sub_histo);
}

// ---------------------------------------------------------------------------
// final: combine the three regions into the 8-bit output histogram
// ---------------------------------------------------------------------------

__kernel void histo_final_kernel (
    unsigned int sm_range_min,
    unsigned int sm_range_max,
    unsigned int histo_height,
    unsigned int histo_width,
    __global unsigned int *global_subhisto,
    __global unsigned short *global_histo,
    __global unsigned int *global_overflow,
    __global unsigned char *final_histo)
{
  unsigned int blockDimx = get_local_size(0);
  unsigned int gridDimx = get_num_groups(0);
  unsigned int start_offset = get_local_id(0) + get_group_id(0) * blockDimx;

  unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
  unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;
  unsigned int total_bins = histo_height * histo_width;
  unsigned int step = gridDimx * blockDimx;

  /* Clear the lower region of the global histogram into the output. */
  for (unsigned int i = start_offset; i < size_low_histo; i += step)
  {
    unsigned int v = global_histo[i];
    global_histo[i] = 0;
    final_histo[i] = (unsigned char) (v < 255 ? v : 255);
  }

  /* Clear the middle region of the overflow buffer and write sub-histogram
   * counts (upstream ignores the overflow counters here). */
  for (unsigned int i = size_low_histo + start_offset;
       i < size_low_histo + size_mid_histo; i += step)
  {
    global_overflow[i] = 0;
    unsigned int v = global_subhisto[i];
    final_histo[i] = (unsigned char) (v < 255 ? v : 255);
  }

  /* Clear the upper region of the global histogram. */
  for (unsigned int i = size_low_histo + size_mid_histo + start_offset;
       i < total_bins; i += step)
  {
    unsigned int v = global_histo[i];
    global_histo[i] = 0;
    final_histo[i] = (unsigned char) (v < 255 ? v : 255);
  }
}