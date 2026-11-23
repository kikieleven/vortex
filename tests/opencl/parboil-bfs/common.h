#ifndef COMMON_H
#define COMMON_H

/* Parboil bfs src/opencl_base/config.h, with the work-group size reduced to
 * Vortex's cap of 16 work-items. LOCAL_MEM_SIZE bounds the per-group frontier
 * queue; upstream notes it must be adjusted for high-degree graphs, and 1024
 * ints is ample for 16 threads x a few edges each. */
#define MAX_THREADS_PER_BLOCK 16
#define LOCAL_MEM_SIZE 1024

#define INF 2147483647
#define UP_LIMIT 16677216
#define WHITE 16677217
#define GRAY 16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221

struct Node {
  int x;
  int y;
};

struct Edge {
  int x;
  int y;
};

#endif