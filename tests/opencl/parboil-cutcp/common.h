#ifndef COMMON_H
#define COMMON_H

/* Parboil cutcp src/opencl_base/macros.h. */
#define NBRLIST_DIM 11
#define NBRLIST_MAXLEN (NBRLIST_DIM * NBRLIST_DIM * NBRLIST_DIM)

#define BIN_DEPTH 8
#define BIN_SIZE 32
#define BIN_CACHE_MAXLEN 32

#define BIN_LENGTH 4.0f
#define BIN_INVLEN (1.0f / BIN_LENGTH)

#define REGION_SIZE 512
#define SUB_REGION_SIZE 128

#endif