// Blue Porcelain basic/NBODY — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/NBODY/n_body.cu
// One work-group per body; work-items split that body's force accumulation over
// a range of the other bodies, then a local-memory tree reduction produces the
// resultant force.
//
// Adaptations for Vortex:
//   - work-groups are 16 wide instead of 256, so bodiesPerThread =
//     ceil(bodyCount/16). The mapping `tid = localId * bodiesPerThread` is the
//     upstream one.
//   - upstream puts __syncthreads() inside `if (tid < bodyCount)`, a divergent
//     barrier; here every work-item always reaches the barrier and the bound is
//     applied to the accumulation loop only.
//   - positions are float4 (x, y, z, unused) instead of a struct, which keeps
//     the three-component arithmetic identical and the indexing simple.

#define GROUP_SIZE 16

__kernel void force_comp(__global const float4 *positions,
                         int body_count,
                         __global float *resultant_force,
                         int bodies_per_thread)
{
  __local float cache[GROUP_SIZE];

  int lid = get_local_id(0);
  int home = get_group_id(0);

  float acc = 0.0f;
  int tid = lid * bodies_per_thread;
  int limit = tid + bodies_per_thread;

  for (; tid < limit && tid < body_count; ++tid) {
    if (home != tid) {
      float4 p = positions[home];
      float4 q = positions[tid];
      float dx = p.x - q.x;
      float dy = p.y - q.y;
      float dz = p.z - q.z;
      acc += 1.0f / sqrt(dx * dx + dy * dy + dz * dz);
    }
  }

  cache[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
    if (lid < s)
      cache[lid] += cache[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0)
    resultant_force[home] = cache[0];
}