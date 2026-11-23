// Parboil sad (src/opencl_base/sad_kernel.cl) — H.264 flexible block-size
// SAD (sum of absolute differences) motion estimation.
//
// Ported faithfully: the 4x4 SAD search, the VBSME 8x4/4x8/8x8/8x16/16x8/16x16
// combine kernels, the block-type layouts and the search-position assignment
// are unchanged.
//
// Adaptations for Vortex:
//   - the reference frame is a plain __global unsigned short buffer instead of
//     an OpenCL image2d (the upstream sampler is CLK_ADDRESS_CLAMP_TO_EDGE;
//     ref_at() reproduces that clamp explicitly). This keeps the case runnable
//     without the image-enabled PoCL-Vortex build; the pixel math is identical.
//   - upstream launches 61-work-item and 32x4-work-item blocks. Vortex caps a
//     work-group at 16, so each physical group of 16 work-items iterates over
//     the upstream logical thread indices (logical < 61 / 128 / 32).
//   - ushort2 vector loads/stores are scalarised (same elements, no vector
//     codegen dependency); mul24 is written as a plain multiply.

#include "common.h"

#define GROUP_SIZE 16

/* CLK_ADDRESS_CLAMP_TO_EDGE, channel CL_UNSIGNED_INT16, CL_FILTER_NEAREST. */
inline unsigned short ref_at(__global const unsigned short *ref,
                             int x, int y, int width, int height) {
  if (x < 0) x = 0;
  else if (x >= width) x = width - 1;
  if (y < 0) y = 0;
  else if (y >= height) y = height - 1;
  return ref[y * width + x];
}

__kernel void mb_sad_calc(__global unsigned short *blk_sad,
                          __global unsigned short *frame,
                          int mb_width,
                          int mb_height,
                          __global const unsigned short *ref,
                          int ref_width,
                          int ref_height)
{
  /* logical thread index within the upstream 61-work-item block; the launch
   * uses get_local_size(0) groups of work-items for one macroblock cell */
  int lt = get_local_id(0) + get_local_size(0) * get_group_id(0);
  if (lt >= CEIL_POS)
    return;

  int tx = (lt / CEIL_POS) % THREADS_W;
  int ty = (lt / CEIL_POS) / THREADS_W;
  int bx = get_group_id(1);
  int by = get_group_id(2);
  int img_width = mb_width*16;

  // Macroblock and sub-block coordinates
  int mb_x = (tx + bx * THREADS_W) >> 2;
  int mb_y = (ty + by * THREADS_H) >> 2;
  int block_x = (tx + bx * THREADS_W) & 0x03;
  int block_y = (ty + by * THREADS_H) & 0x03;

  if ((mb_x < mb_width) && (mb_y < mb_height))
    {
      // Pixel offset of the origin of the current 4x4 block
      int frame_x = ((mb_x << 2) + block_x) << 2;
      int frame_y = ((mb_y << 2) + block_y) << 2;

      // Origin of the search area for this 4x4 block
      int ref_x = frame_x - SEARCH_RANGE;
      int ref_y = frame_y - SEARCH_RANGE;

      // Origin in the current frame for this 4x4 block
      int cur_o = frame_y * img_width + frame_x;

      int search_pos;
      int search_pos_base = (lt % CEIL_POS) * POS_PER_THREAD;
      int search_pos_end = search_pos_base + POS_PER_THREAD;

      blk_sad += mb_width * mb_height * MAX_POS_PADDED * (9 + 16) +
        (mb_y * mb_width + mb_x) * MAX_POS_PADDED * 16 +
        (4 * block_y + block_x) * MAX_POS_PADDED;

      if (search_pos_end > MAX_POS)
        search_pos_end = MAX_POS;

      for (search_pos = search_pos_base;
           search_pos < search_pos_end;
           search_pos++) {
        unsigned short sad4x4 = 0;
        int search_off_x = ref_x + (search_pos % SEARCH_DIMENSION);
        int search_off_y = ref_y + (search_pos / SEARCH_DIMENSION);

        for(int y=0; y<4; y++) {
          for (int x=0; x<4; x++) {
            sad4x4 += abs((int)ref_at(ref, search_off_x + x, search_off_y + y,
                                      ref_width, ref_height) -
                          (int)frame[cur_o + y * img_width + x]);
          }
        }

        blk_sad[search_pos] = sad4x4;
      }
    }
}

__kernel void larger_sad_calc_8(__global unsigned short *blk_sad,
                                int mb_width,
                                int mb_height)
{
  int mb_x = get_group_id(0);
  int mb_y = get_group_id(1);

  int macroblocks = mb_width * mb_height;
  int macroblock_index = (mb_y * mb_width + mb_x) * MAX_POS_PADDED;

  /* upstream block is 32 x 4 logical work-items; logical thread (l0,l1)
   * strides over search positions by 32 */
  for (int logical = get_local_id(0); logical < 32 * 4;
       logical += get_local_size(0)) {
    int l0 = logical % 32;
    int l1 = logical / 32;
    int tx = l1 & 1;
    int ty = l1 >> 1;

    __global unsigned short *bi;
    __global unsigned short *bo_6, *bo_5, *bo_4;

    bi = blk_sad
      + (macroblocks * 25 + (ty * 8 + tx * 2)) * MAX_POS_PADDED
      + macroblock_index * 16;

    bo_6 = blk_sad
      + ((macroblocks << 4) + macroblocks + (ty * 4 + tx * 2)) * MAX_POS_PADDED
      + macroblock_index * 8;

    bo_5 = blk_sad
      + ((macroblocks << 3) + macroblocks + (ty * 4 + tx)) * MAX_POS_PADDED
      + macroblock_index * 8;

    bo_4 = blk_sad
      + ((macroblocks << 2) + macroblocks + (ty * 2 + tx)) * MAX_POS_PADDED
      + macroblock_index * 4;

    for (int search_pos = l0; search_pos < (MAX_POS+1)/2;
         search_pos += 32) {
      unsigned short s00x = bi[search_pos*2];
      unsigned short s00y = bi[search_pos*2+1];
      unsigned short s01x = bi[(search_pos + MAX_POS_PADDED/2)*2];
      unsigned short s01y = bi[(search_pos + MAX_POS_PADDED/2)*2+1];
      unsigned short s10x = bi[(search_pos + 4*MAX_POS_PADDED/2)*2];
      unsigned short s10y = bi[(search_pos + 4*MAX_POS_PADDED/2)*2+1];
      unsigned short s11x = bi[(search_pos + 5*MAX_POS_PADDED/2)*2];
      unsigned short s11y = bi[(search_pos + 5*MAX_POS_PADDED/2)*2+1];

      bo_6[search_pos*2] = s00x + s10x;
      bo_6[search_pos*2+1] = s00y + s10y;
      bo_6[(search_pos+MAX_POS_PADDED/2)*2] = s01x + s11x;
      bo_6[(search_pos+MAX_POS_PADDED/2)*2+1] = s01y + s11y;
      bo_5[search_pos*2] = s00x + s01x;
      bo_5[search_pos*2+1] = s00y + s01y;
      bo_5[(search_pos+2*MAX_POS_PADDED/2)*2] = s10x + s11x;
      bo_5[(search_pos+2*MAX_POS_PADDED/2)*2+1] = s10y + s11y;
      bo_4[search_pos*2] = (s00x + s01x) + (s10x + s11x);
      bo_4[search_pos*2+1] = (s00y + s01y) + (s10y + s11y);
    }
  }
}

__kernel void larger_sad_calc_16(__global unsigned short *blk_sad,
                                 int mb_width,
                                 int mb_height)
{
  int mb_x = get_group_id(0);
  int mb_y = get_group_id(1);

  int macroblocks = mb_width * mb_height * MAX_POS_PADDED;
  int macroblock_index = (mb_y * mb_width + mb_x) * MAX_POS_PADDED;

  __global unsigned short *bi;
  __global unsigned short *bo_3, *bo_2, *bo_1;

  bi = blk_sad + ((macroblocks + macroblock_index) << 2) + macroblocks;

  bo_3 = blk_sad + ((macroblocks + macroblock_index) << 1) + macroblocks;

  bo_2 = blk_sad + macroblocks + macroblock_index * 2;

  bo_1 = blk_sad + macroblock_index;

  /* upstream block is 32 x 1 logical work-items */
  for (int search_pos = get_local_id(0); search_pos < (MAX_POS+1)/2;
       search_pos += get_local_size(0)) {
    unsigned short s00x = bi[search_pos*2];
    unsigned short s00y = bi[search_pos*2+1];
    unsigned short s01x = bi[(search_pos + MAX_POS_PADDED/2)*2];
    unsigned short s01y = bi[(search_pos + MAX_POS_PADDED/2)*2+1];
    unsigned short s10x = bi[(search_pos + 2*MAX_POS_PADDED/2)*2];
    unsigned short s10y = bi[(search_pos + 2*MAX_POS_PADDED/2)*2+1];
    unsigned short s11x = bi[(search_pos + 3*MAX_POS_PADDED/2)*2];
    unsigned short s11y = bi[(search_pos + 3*MAX_POS_PADDED/2)*2+1];

    bo_3[search_pos*2] = s00x + s10x;
    bo_3[search_pos*2+1] = s00y + s10y;
    bo_3[(search_pos+MAX_POS_PADDED/2)*2] = s01x + s11x;
    bo_3[(search_pos+MAX_POS_PADDED/2)*2+1] = s01y + s11y;
    bo_2[search_pos*2] = s00x + s01x;
    bo_2[search_pos*2+1] = s00y + s01y;
    bo_2[(search_pos+MAX_POS_PADDED/2)*2] = s10x + s11x;
    bo_2[(search_pos+MAX_POS_PADDED/2)*2+1] = s10y + s11y;
    bo_1[search_pos*2] = (s00x + s01x) + (s10x + s11x);
    bo_1[search_pos*2+1] = (s00y + s01y) + (s10y + s11y);
  }
}