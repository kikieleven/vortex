// Parboil mri-gridding (src/opencl_base/GPU_kernels.cl, sort.cl,
// scanLargeArray.cl) — non-uniform FFT gridding: bin the k-space samples,
// sort them by bin, build the bin start addresses with a prefix scan, and
// gather each grid point's contributions from the neighbouring bins.
//
// Ported faithfully: the binning key/overflow rule, the reorder, the gridding
// gather (bin neighbourhood, Kaiser-Bessel kernel polynomial, density
// estimate) and the scan kernels are unchanged.
//
// Adaptations for Vortex:
//   - the gridding kernel is launched with a 4x2x2 = 16-work-item group
//     (Vortex's cap); the local sample tile (TILE=64) is loaded by looping the
//     flat index instead of the single-if copy.
//   - the runtime grid constants are kernel arguments instead of -D options,
//     and float2 grid data is a flat float array (real, imag).
//   - native_divide is written as a plain division (same value).
//   - the scan kernels use a 16-work-item block and plain local indexing; the
//     upstream conflict-free offsets are zero at these indices.
//   - the radix-sort stage is not run on the device: the host performs a stable
//     sort of the (key,value) pairs produced by binning_kernel (see main.cc).
//     The sorted sample order and the scan input are identical; the sort
//     kernels are therefore not covered by this test.

#include "common.h"

__kernel void binning_kernel (unsigned int n,
                              __global struct ReconstructionSample* sample_g,
                              __global unsigned int* idxKey_g,
                              __global unsigned int* idxValue_g,
                              __global unsigned int* binCount_g,
                              unsigned int binsize, unsigned int gridNumElems,
                              int size_xy, int gridsize1)
{
  unsigned int key;
  unsigned int sampleIdx = get_global_id(0);
  struct ReconstructionSample pt;
  unsigned int binIdx;
  unsigned int count;

  if (sampleIdx < n){
    pt = sample_g[sampleIdx];

    binIdx = (unsigned int)(pt.kZ)*size_xy + (unsigned int)(pt.kY)*gridsize1 + (unsigned int)(pt.kX);

    count = atomic_add(binCount_g+binIdx, 1);
    if (count < binsize){
      key = binIdx;
    } else {
      atomic_sub(binCount_g+binIdx, 1);
      key = gridNumElems;
    }

    idxKey_g[sampleIdx] = key;
    idxValue_g[sampleIdx] = sampleIdx;
  }
}

__kernel void reorder_kernel(int n,
                             __global unsigned int* idxValue_g,
                             __global struct ReconstructionSample* samples_g,
                             __global struct ReconstructionSample* sortedSample_g){
  unsigned int index = get_global_id(0);
  unsigned int old_index;
  struct ReconstructionSample pt;

  if (index < n){
    old_index = idxValue_g[index];
    pt = samples_g[old_index];
    sortedSample_g[index] = pt;
  }
}

float kernel_value(float v){
  float rValue = 0;
  float z = v*v;

  float num = (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z* (z*
                (z* 0.210580722890567e-22f  + 0.380715242345326e-19f ) +
                 0.479440257548300e-16f) + 0.435125971262668e-13f ) +
                 0.300931127112960e-10f) + 0.160224679395361e-7f  ) +
                 0.654858370096785e-5f)  + 0.202591084143397e-2f  ) +
                 0.463076284721000e0f)   + 0.754337328948189e2f   ) +
                 0.830792541809429e4f)   + 0.571661130563785e6f   ) +
                 0.216415572361227e8f)   + 0.356644482244025e9f  ) +
                 0.144048298227235e10f);

  float den = (z*(z*(z-0.307646912682801e4f)+0.347626332405882e7f)-0.144048298227235e10f);

  rValue = -num/den;
  return rValue;
}

__kernel void gridding_GPU (__global const struct ReconstructionSample* sample_g,
                            __global const unsigned int* binStartAddr_g,
                            __global float* gridData_g,   /* 2 floats per point */
                            __global float* sampleDensity_g,
                            float beta,
                            int gsize1, int gsize2, int gsize3,
                            int size_xy,
                            float cutoff2, float ceil_cutoff,
                            float one_over_cutoff2,
                            int dims0, int dims1, int dims2)
{
  __local struct ReconstructionSample sharedBin[TILE];

  const int ls0 = get_local_size(0);
  const int ls1 = get_local_size(1);
  const int ls2 = get_local_size(2);
  const int lid0 = get_local_id(0);
  const int lid1 = get_local_id(1);
  const int lid2 = get_local_id(2);
  const int flatIdx = lid2*ls1*ls0 + lid1*ls0 + lid0;
  const int group_1 = get_group_id(1);

  const int z0 = ls2*(group_1/(gsize2/ls1));
  const int y0 = ls1*(group_1%(gsize2/ls1));
  const int x0 = get_group_id(0)*ls0;

  const int X  = x0+lid0;
  const int Y  = y0+lid1;
  const int Z  = z0+lid2;

  const int xl = x0-(int)ceil_cutoff;
  const int xL = (xl < 0) ? 0 : xl;
  const int xh = x0+ls0+(int)ceil_cutoff;
  const int xH = (xh >= gsize1) ? gsize1-1 : xh;

  const int yl = y0-(int)ceil_cutoff;
  const int yL = (yl < 0) ? 0 : yl;
  const int yh = y0+ls1+(int)ceil_cutoff;
  const int yH = (yh >= gsize2) ? gsize2-1 : yh;

  const int zl = z0-(int)ceil_cutoff;
  const int zL = (zl < 0) ? 0 : zl;
  const int zh = z0+ls2+(int)ceil_cutoff;
  const int zH = (zh >= gsize3) ? gsize3-1 : zh;

  const int idx = Z*size_xy + Y*gsize1 + X;

  float ptre = 0.0f;
  float pti = 0.0f;
  float density = 0.0f;

  for (int z = zL; z <= zH; z++){
    for (int y = yL; y <= yH; y++){
      __global const unsigned int *addr = binStartAddr_g+z*size_xy+ y*gsize1;
      const unsigned int start = *(addr+xL);
      const unsigned int end   = *(addr+xH+1);
      const unsigned int delta = end-start;
      for (int x = 0; x < ((delta+TILE-1)>>LOG_TILE); x++){
        int tileSize = ((delta-(x<<LOG_TILE)) > TILE) ? TILE : (delta-(x<<LOG_TILE));
        int globalIdx = flatIdx+(x<<LOG_TILE);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = flatIdx; j < tileSize; j += ls0*ls1*ls2) {
          sharedBin[j] = sample_g[start+j+(x<<LOG_TILE)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j=0; j< tileSize; j++){
          const float real = sharedBin[j].real;
          const float imag = sharedBin[j].imag;
          const float sdc = sharedBin[j].sdc;

          if((real != 0.0f || imag != 0.0f) && sdc != 0.0f){
            float v = (sharedBin[j].kX-X)*(sharedBin[j].kX-X);
            v += (sharedBin[j].kY-Y)*(sharedBin[j].kY-Y);
            v += (sharedBin[j].kZ-Z)*(sharedBin[j].kZ-Z);
            if(v<cutoff2){
               const float w = kernel_value(beta*sqrt(1.0f-(v*one_over_cutoff2))) *sdc;
              ptre += w*real;
              pti += w*imag;
              density += 1.0f;
            }
          }
        }
      }
    }
  }

  gridData_g[2*idx] = ptre;
  gridData_g[2*idx+1] = pti;
  sampleDensity_g[idx] = density;
}

// ---------------------------------------------------------------------------
// scanLargeArray.cl
// ---------------------------------------------------------------------------

__kernel void scan_L1_kernel(unsigned int n, __global unsigned int* dataBase,
                             unsigned int data_offset,
                             __global unsigned int* interBase,
                             unsigned int inter_offset)
{
  __local unsigned int s_data[SCAN_LOCAL_SIZE];

  __global unsigned int *data = dataBase + data_offset;
  __global unsigned int *inter = interBase + inter_offset;

  unsigned int thid = get_local_id(0);
  unsigned int g_ai = get_group_id(0)*2*get_local_size(0) + get_local_id(0);
  unsigned int g_bi = g_ai + get_local_size(0);

  unsigned int s_ai = thid + CONFLICT_FREE_OFFSET(thid);
  unsigned int s_bi = thid + get_local_size(0) + CONFLICT_FREE_OFFSET(thid + get_local_size(0));

  s_data[s_ai] = (g_ai < n) ? data[g_ai] : 0;
  s_data[s_bi] = (g_bi < n) ? data[g_bi] : 0;

  unsigned int stride = 1;
  for (unsigned int d = get_local_size(0); d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      s_data[bi] += s_data[ai];
    }
    stride *= 2;
  }

  if (thid == 0) {
    unsigned int last = get_local_size(0)*2 -1;
    last += CONFLICT_FREE_OFFSET(last);
    inter[get_group_id(0)] = s_data[last];
    s_data[last] = 0;
  }

  for (unsigned int d = 1; d <= get_local_size(0); d *= 2) {
    stride >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      unsigned int t  = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (g_ai < n) { data[g_ai] = s_data[s_ai]; }
  if (g_bi < n) { data[g_bi] = s_data[s_bi]; }
}

__kernel void scan_inter1_kernel(__global unsigned int* data, unsigned int iter)
{
  __local unsigned int s_data[SCAN_LOCAL_SIZE];

  unsigned int thid = get_local_id(0);
  unsigned int gthid = get_global_id(0);
  unsigned int gi = 2*iter*gthid;
  unsigned int g_ai = gi + iter - 1;
  unsigned int g_bi = g_ai + iter;

  unsigned int s_ai = 2*thid + CONFLICT_FREE_OFFSET(2*thid);
  unsigned int s_bi = 2*thid + 1 + CONFLICT_FREE_OFFSET(2*thid + 1);

  s_data[s_ai] = data[g_ai];
  s_data[s_bi] = data[g_bi];

  unsigned int stride = 1;
  for (unsigned int d = get_local_size(0); d > 0; d >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      s_data[bi] += s_data[ai];
    }
    stride *= 2;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  data[g_ai] = s_data[s_ai];
  data[g_bi] = s_data[s_bi];
}

__kernel void scan_inter2_kernel(__global unsigned int* data, unsigned int iter)
{
  __local unsigned int s_data[SCAN_LOCAL_SIZE];

  unsigned int thid = get_local_id(0);
  unsigned int gthid = get_global_id(0);
  unsigned int gi = 2*iter*gthid;
  unsigned int g_ai = gi + iter - 1;
  unsigned int g_bi = g_ai + iter;

  unsigned int s_ai = 2*thid + CONFLICT_FREE_OFFSET(2*thid);
  unsigned int s_bi = 2*thid + 1 + CONFLICT_FREE_OFFSET(2*thid + 1);

  s_data[s_ai] = data[g_ai];
  s_data[s_bi] = data[g_bi];

  unsigned int stride = get_local_size(0)*2;
  for (unsigned int d = 1; d <= get_local_size(0); d *= 2) {
    stride >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (thid < d) {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);
      unsigned int t  = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  data[g_ai] = s_data[s_ai];
  data[g_bi] = s_data[s_bi];
}

__kernel void uniformAdd(unsigned int n, __global unsigned int *dataBase,
                         unsigned int data_offset,
                         __global unsigned int *interBase,
                         unsigned int inter_offset)
{
  __local unsigned int uni;

  __global unsigned int *data = dataBase + data_offset;
  __global unsigned int *inter = interBase + inter_offset;

  if (get_local_id(0) == 0) { uni = inter[get_group_id(0)]; }
  barrier(CLK_LOCAL_MEM_FENCE);

  unsigned int g_ai = get_group_id(0)*2*get_local_size(0) + get_local_id(0);
  unsigned int g_bi = g_ai + get_local_size(0);

  if (g_ai < n) { data[g_ai] += uni; }
  if (g_bi < n) { data[g_bi] += uni; }
}