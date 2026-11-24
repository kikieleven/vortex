#ifndef COMMON_H
#define COMMON_H

/* Parboil mri-gridding src/opencl_base/UDTypes.h + kernel constants. */
struct ReconstructionSample {
  float real;
  float imag;
  float kX;
  float kY;
  float kZ;
  float sdc;
};

#define TILE 64
#define LOG_TILE 6

/* scanLargeArray.cl: bank-conflict-free padding. For the indices used with a
 * 16-work-item block (all < 64) the offset is zero, so the expression is kept
 * for fidelity but the local arrays are plain. */
#define LNB 4
#define CONFLICT_FREE_OFFSET(index) \
  (((unsigned int)(index) >> (((LNB + (index)) < (32 - 2*LNB)) ? (LNB + (index)) : (32 - 2*LNB))) >> (2*LNB))

#define SCAN_LOCAL_SIZE 64

#endif