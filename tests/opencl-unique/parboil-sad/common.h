#ifndef COMMON_H
#define COMMON_H

/* Parboil sad src/opencl_base/sad.h + sad_kernel.h. */
#define SEARCH_RANGE 16
#define SEARCH_DIMENSION (2*SEARCH_RANGE+1)
#define MAX_POS 1089
#define MAX_POS_PADDED 1096

/* sad_kernel.h: the optimized kernel requires POS_PER_THREAD % 3 == 0 and
 * POS_PER_THREAD <= 33. */
#define POS_PER_THREAD 18
#define CEIL_POS ((MAX_POS + POS_PER_THREAD - 1) / POS_PER_THREAD)

#define THREADS_W 1
#define THREADS_H 1

/* VBSME block-type offsets (sad.h). */
#define SAD_TYPE_1_IX(image_size) 0
#define SAD_TYPE_2_IX(image_size) ((image_size)*MAX_POS_PADDED)
#define SAD_TYPE_3_IX(image_size) ((image_size)*(3*MAX_POS_PADDED))
#define SAD_TYPE_4_IX(image_size) ((image_size)*(5*MAX_POS_PADDED))
#define SAD_TYPE_5_IX(image_size) ((image_size)*(9*MAX_POS_PADDED))
#define SAD_TYPE_6_IX(image_size) ((image_size)*(17*MAX_POS_PADDED))
#define SAD_TYPE_7_IX(image_size) ((image_size)*(25*MAX_POS_PADDED))

#define SAD_TYPE_1_CT 1
#define SAD_TYPE_2_CT 2
#define SAD_TYPE_3_CT 2
#define SAD_TYPE_4_CT 4
#define SAD_TYPE_5_CT 8
#define SAD_TYPE_6_CT 8
#define SAD_TYPE_7_CT 16

#endif