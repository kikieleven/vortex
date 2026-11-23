// Parboil bfs (src/opencl_base/kernel.cl) — level-synchronous BFS with a
// block-local frontier queue and global atomics.
//
// Ported faithfully: the frontier propagation, the gray/BLACK colouring, the
// atom_min cost update, the per-block local queue and the block-shift
// compaction are unchanged. `atom_*` (OpenCL 1.1) is written with the
// equivalent core atomic_* builtins; the extension pragmas are not needed
// because PoCL exposes 32-bit atomics by default. Atomics require the Vortex A
// extension (see ci/testcases/opencl.yaml).
//
// Adaptation: MAX_THREADS_PER_BLOCK is 16 (Vortex's work-group cap) instead of
// 256. The kernel already indexes by global id, so only the host launch
// geometry changes.

#include "common.h"

__kernel void
BFS_kernel(__global int *q1,
           __global int *q2,
           __global struct Node *g_graph_nodes,
           __global struct Edge *g_graph_edges,
           __global int *g_color,
           __global int *g_cost,
           __global int *tail,
           int no_of_nodes,
           int gray_shade,
           int k ,
           __local int *local_q_tail,
           __local int *local_q,
           __local int *shift)
{

  if(get_local_id(0) == 0){
    *local_q_tail = 0;//initialize the tail of w-queue
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  //first, propagate and add the new frontier elements into w-queues
  int tid = get_global_id(0);

  if( tid<no_of_nodes)
  {
    int pid = q1[tid]; //the current frontier node, or the parent node of the new frontier nodes
    g_color[pid] = BLACK;
    int cur_cost = g_cost[pid];
    //into
    struct Node cur_node = g_graph_nodes[pid];

    for(int i=cur_node.x; i<cur_node.y + cur_node.x; i++)//visit each neighbor of the
      //current frontier node.
    {
      struct Edge cur_edge = g_graph_edges[i];
      int id = cur_edge.x;
      int cost = cur_edge.y;
      cost += cur_cost;
      int orig_cost = atomic_min (&g_cost[id],cost);
      if(orig_cost > cost){//the node should be visited
        if(g_color[id] > UP_LIMIT){
         int old_color = atomic_xchg (&g_color[id],gray_shade);
          //this guarantees that only one thread will push this node
          //into a queue
          if(old_color != gray_shade) {
            //atomic operation guarantees the correctness
            //even if multiple warps are executing simultaneously
            int index = atomic_add (local_q_tail,1);
            local_q[index] = id;
          }
        }
      }
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  if(get_local_id(0) == 0){
    int tot_sum = *local_q_tail;
    //the offset or "shift" of the block-level queue within the grid-level queue
    //is determined by atomic operation
    *shift = atomic_add (tail,tot_sum);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  //shift within a w-queue
  int local_shift = get_local_id(0);

  while(local_shift < *local_q_tail){
    q2[*shift + local_shift] = local_q[local_shift];
    //multiple threads are copying elements at the same time,
    //so we shift by multiple elements for next iteration
    local_shift += get_local_size(0);
  }
}