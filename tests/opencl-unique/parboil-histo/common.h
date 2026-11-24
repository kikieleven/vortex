#ifndef COMMON_H
#define COMMON_H

/* Parboil histo src/opencl_base/main.cpp compile options and kernel
 * constants. KB is set to 8 because Vortex's local memory is 16 KiB:
 * KB*256 unsigned ints = 8 KiB of shared memory per work-group.
 * BINS_PER_BLOCK = KB*1024 = 8192.
 */
#define KB 8
#define BINS_PER_BLOCK (KB * 1024)
#define UNROLL 16
#define PRESCAN_THREADS 16

/* Work-group size (Vortex's cap) and the number of scanning groups used by
 * the host launches. */
#define BLOCK_SIZE 16
#define MAIN_GROUPS 4
#define PRESCAN_GROUPS 8
#define FINAL_GROUPS 16

#endif