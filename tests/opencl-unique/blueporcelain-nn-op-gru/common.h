#ifndef COMMON_H
#define COMMON_H

#ifndef TYPE
#define TYPE float
#endif

#define GRU_SEQ    4
#define GRU_IN     8
#define GRU_HIDDEN 8
#define GRU_GATES  3   /* r, z, n in PyTorch's packed order */

#endif // COMMON_H
