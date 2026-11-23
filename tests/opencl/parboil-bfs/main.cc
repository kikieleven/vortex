// Parboil bfs — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/parboil/benchmarks/bfs/src/opencl_base/kernel.cl
//   third_benchmark/parboil/benchmarks/bfs/src/opencl_base/main.cpp
//     (graph file layout, frontier loop, tail handshake, colour shades)
//
// Parboil reads a graph file and writes the cost vector; this port generates a
// deterministic near-regular graph and compares the device costs against a
// host BFS.
//
// Graph: every node links to a fixed set of neighbours derived from its index
// (the (i+1) edge makes the graph a ring, so all nodes are reachable from the
// source). All edge costs are 1, which is the case Parboil's input graphs use;
// the kernel's atom_min relaxation is then exactly BFS distance.
//
// Adaptation: upstream uses 256-work-item groups and a grid sized from the
// frontier length. Vortex caps a work-group at 16, so groups are always 16 and
// the grid is ceil(frontier/16)*16; the kernel's tid guard skips idle
// work-items.
//
// CI: atomics need the RVA A extension, so this case is in EXCLUDE and runs
// via its opencl.yaml blackbox entry with -DVX_CFG_EXT_A_ENABLE.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <queue>

#include "common.h"

#define KERNEL_NAME "BFS_kernel"

#define LWS 16

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     cleanup();                                                        \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       cleanup();                                                      \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }

  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem nodes_memobj = NULL;
cl_mem edges_memobj = NULL;
cl_mem color_memobj = NULL;
cl_mem cost_memobj = NULL;
cl_mem q1_memobj = NULL;
cl_mem q2_memobj = NULL;
cl_mem tail_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (nodes_memobj) clReleaseMemObject(nodes_memobj);
  if (edges_memobj) clReleaseMemObject(edges_memobj);
  if (color_memobj) clReleaseMemObject(color_memobj);
  if (cost_memobj) clReleaseMemObject(cost_memobj);
  if (q1_memobj) clReleaseMemObject(q1_memobj);
  if (q2_memobj) clReleaseMemObject(q2_memobj);
  if (tail_memobj) clReleaseMemObject(tail_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 256;

static void show_usage() {
  printf("Usage: [-n number of nodes] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 2) {
    printf("Error: size must be at least 2\n");
    return -1;
  }

  const int n = size;
  const int source = 0;

  printf("Graph nodes=%d\n", n);

  // Build a deterministic near-regular graph. Every node gets the (i+1)%n
  // edge, which makes the graph a ring, plus a few index-derived neighbours;
  // duplicates and self loops are dropped.
  std::vector<struct Node> h_nodes(n);
  std::vector<struct Edge> h_edges;
  int edge_start = 0;
  for (int i = 0; i < n; ++i) {
    h_nodes[i].x = edge_start;
    int candidates[4] = {
      (i + 1) % n,
      (i + 5) % n,
      (i + 13) % n,
      (i * 7 + 3) % n
    };
    int count = 0;
    for (int c = 0; c < 4; ++c) {
      int cand = candidates[c];
      if (cand == i)
        continue;
      bool dup = false;
      for (int p = 0; p < c; ++p)
        if (candidates[p] == cand)
          dup = true;
      if (dup)
        continue;
      struct Edge e;
      e.x = cand;
      e.y = 1;   // unit edge cost, as in Parboil's input graphs
      h_edges.push_back(e);
      ++count;
    }
    h_nodes[i].y = count;
    edge_start += count;
  }
  const int num_edges = (int)h_edges.size();

  std::vector<int> h_color(n, WHITE);
  std::vector<int> h_cost(n, INF);
  h_cost[source] = 0;

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  nodes_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(struct Node), NULL, &_err));
  edges_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, num_edges * sizeof(struct Edge), NULL, &_err));
  color_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &_err));
  cost_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &_err));
  q1_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &_err));
  q2_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &_err));
  tail_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &_err));

  printf("Create program from kernel source\n");
  size_t kernel_size = 0;
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, nodes_memobj, CL_TRUE, 0, n * sizeof(struct Node), h_nodes.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, edges_memobj, CL_TRUE, 0, num_edges * sizeof(struct Edge), h_edges.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, color_memobj, CL_TRUE, 0, n * sizeof(int), h_color.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, cost_memobj, CL_TRUE, 0, n * sizeof(int), h_cost.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();

  const int h_top = 1;
  const int zero = 0;
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, tail_memobj, CL_TRUE, 0, sizeof(int), &h_top, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, q1_memobj, CL_TRUE, 0, sizeof(int), &source, 0, NULL, NULL));

  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&nodes_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&edges_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&color_memobj));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&cost_memobj));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&tail_memobj));

  int levels = 0;
  int num_t;
  do {
    CL_CHECK(clEnqueueReadBuffer(commandQueue, tail_memobj, CL_TRUE, 0, sizeof(int), &num_t, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, tail_memobj, CL_TRUE, 0, sizeof(int), &zero, 0, NULL, NULL));

    if (num_t == 0)
      break;

    const size_t grid = (size_t)((num_t + LWS - 1) / LWS) * LWS;
    const size_t block = LWS;

    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(int), (void *)&num_t));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), (void *)&levels));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), NULL));
    CL_CHECK(clSetKernelArg(kernel, 11, LOCAL_MEM_SIZE * sizeof(int), NULL));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int), NULL));

    int gray;
    if (levels % 2 == 0) {
      gray = GRAY0;
      CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&q1_memobj));
      CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&q2_memobj));
    } else {
      gray = GRAY1;
      CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&q2_memobj));
      CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&q1_memobj));
    }
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), (void *)&gray));

    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    &grid, &block, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    ++levels;
  } while (true);

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms (BFS levels=%d)\n", elapsed, levels);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, cost_memobj, CL_TRUE, 0, n * sizeof(int),
                               h_cost.data(), 0, NULL, NULL));

  printf("Verify result\n");
  // Host BFS: unit edge costs make the kernel's relaxation identical to the
  // shortest-path distance from the source.
  std::vector<int> ref_cost(n, INF);
  std::vector<int> dist(n, -1);
  std::queue<int> frontier;
  dist[source] = 0;
  frontier.push(source);
  while (!frontier.empty()) {
    int pid = frontier.front();
    frontier.pop();
    for (int e = h_nodes[pid].x; e < h_nodes[pid].x + h_nodes[pid].y; ++e) {
      int id = h_edges[e].x;
      int cost = dist[pid] + h_edges[e].y;
      if (cost < dist[id] || dist[id] < 0) {
        dist[id] = cost;
        frontier.push(id);
      }
    }
  }
  for (int i = 0; i < n; ++i)
    ref_cost[i] = dist[i] < 0 ? INF : dist[i];

  int errors = 0;
  for (int i = 0; i < n; ++i) {
    if (h_cost[i] != ref_cost[i]) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n",
               i, ref_cost[i], h_cost[i]);
      }
      ++errors;
    }
  }

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}