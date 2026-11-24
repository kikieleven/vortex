// Particle filter (single-precision "optimized" variant), ported from Rodinia
// OpenCL: third_benchmark/gpu-rodinia/opencl/particlefilter/particle_single.cl
//
// The upstream double-precision variants (particle_naive.cl,
// particle_double.cl) cannot be ported: the Vortex OpenCL device has no
// cl_khr_fp64 (CL_DEVICE_DOUBLE_FP_CONFIG == 0). This is the float variant.
//
// Vortex mapping: the host launches ONE work-group of LOCAL_SIZE work-items
// with Nparticles == LOCAL_SIZE. That reproduces the upstream CUDA/OpenCL
// single-block assumption: normalize_weights_kernel writes u[0] in work-item 0
// and the other work-items consume it, which is only safe inside a single
// work-group (OpenCL barriers do not synchronize across work-groups).
//
// Arithmetic is kept identical to upstream. Three deliberate fixes:
//   * unsuffixed `50.0` -> `50.0f` (fp64 literals are rejected by the device);
//   * pow(x, 2) -> x*x (identical value, same expression order);
//   * the LCG advance uses unsigned arithmetic so signed overflow is not UB.
// The upstream `return newValue++` rounding helper returns the pre-increment
// value, i.e. it truncates; that behaviour is preserved.

#define SCALE_FACTOR 300

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 16
#endif

#define LCG_A 1103515245u
#define LCG_C 12345u
#define LCG_M 2147483647

static int lcg_next(int s) {
  return (int)(LCG_A * (uint)s + LCG_C);
}

// Uniform in [0,1) from the per-particle seed (advances the seed).
static float d_randu(__global int* seed, int index) {
  int num = lcg_next(seed[index]);
  seed[index] = num % LCG_M;
  return fabs((float)seed[index] / (float)LCG_M);
}

// Box-Muller normal sample.
static float d_randn(__global int* seed, int index) {
  float pi = 3.14159265358979323846f;
  float u = d_randu(seed, index);
  float v = d_randu(seed, index);
  float cosine = cos(2.0f * pi * v);
  float rt = -2.0f * log(u);
  return sqrt(rt) * cosine;
}

// Upstream: `if (value - newValue < .5f) return newValue; else return
// newValue++;` — the post-increment returns the old value, so the fractional
// part is always truncated. Keep that exact behaviour.
static float dev_round_float(float value) {
  int newValue = (int)(value);
  return (float)newValue;
}

static float calc_likelihood_sum(__global const unsigned char* I,
                                 __global const int* ind,
                                 int numOnes, int index) {
  float likelihoodSum = 0.0f;
  for (int x = 0; x < numOnes; ++x) {
    float q = (float)((int)I[ind[index * numOnes + x]] - 100);
    float r = (float)((int)I[ind[index * numOnes + x]] - 228);
    likelihoodSum += (q * q - r * r) / 50.0f;
  }
  return likelihoodSum;
}

// Per-particle likelihood, weight update, and a block sum of the weights.
__kernel void likelihood_kernel(__global float* arrayX,
                                __global float* arrayY,
                                __global float* xj,
                                __global float* yj,
                                __global float* CDF,
                                __global int* ind,
                                __global const int* objxy,
                                __global float* likelihood,
                                __global const unsigned char* I,
                                __global float* u,
                                __global float* weights,
                                int Nparticles,
                                int countOnes,
                                int max_size,
                                int k,
                                int IszY,
                                int Nfr,
                                __global int* seed,
                                __global float* partial_sums) {
  __local float buffer[LOCAL_SIZE];

  const int i = get_global_id(0);
  const int thread_id = get_local_id(0);

  if (i < Nparticles) {
    arrayX[i] = xj[i];
    arrayY[i] = yj[i];
    weights[i] = 1.0f / (float)(Nparticles);

    arrayX[i] = arrayX[i] + 1.0f + 5.0f * d_randn(seed, i);
    arrayY[i] = arrayY[i] - 2.0f + 2.0f * d_randn(seed, i);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  if (i < Nparticles) {
    for (int y = 0; y < countOnes; ++y) {
      int indX = (int)dev_round_float(arrayX[i]) + objxy[y * 2 + 1];
      int indY = (int)dev_round_float(arrayY[i]) + objxy[y * 2];
      ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
      if (ind[i * countOnes + y] >= max_size)
        ind[i * countOnes + y] = 0;
    }
    likelihood[i] = calc_likelihood_sum(I, ind, countOnes, i);
    likelihood[i] = likelihood[i] / (float)(countOnes) - (float)(SCALE_FACTOR);
    weights[i] = weights[i] * exp(likelihood[i]);
  }

  buffer[thread_id] = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (i < Nparticles)
    buffer[thread_id] = weights[i];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int s = LOCAL_SIZE / 2; s > 0; s >>= 1) {
    if (thread_id < s)
      buffer[thread_id] += buffer[thread_id + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (thread_id == 0)
    partial_sums[get_group_id(0)] = buffer[0];
}

// Sum the per-block partial sums into partial_sums[0] (single work-item).
__kernel void sum_kernel(__global float* partial_sums, int Nparticles) {
  if (get_global_id(0) == 0) {
    const int num_blocks = (Nparticles + LOCAL_SIZE - 1) / LOCAL_SIZE;
    float sum = 0.0f;
    for (int x = 0; x < num_blocks; ++x)
      sum += partial_sums[x];
    partial_sums[0] = sum;
  }
}

// Normalize the weights, build the CDF and the stratified sample points u[].
__kernel void normalize_weights_kernel(__global float* weights,
                                       int Nparticles,
                                       __global const float* partial_sums,
                                       __global float* CDF,
                                       __global float* u,
                                       __global int* seed) {
  const int i = get_global_id(0);
  const int local_id = get_local_id(0);

  __local float u1;
  __local float sumWeights;

  if (local_id == 0)
    sumWeights = partial_sums[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (i < Nparticles)
    weights[i] = weights[i] / sumWeights;
  barrier(CLK_GLOBAL_MEM_FENCE);

  if (i == 0) {
    CDF[0] = weights[0];
    for (int x = 1; x < Nparticles; ++x)
      CDF[x] = weights[x] + CDF[x - 1];
    u[0] = (1.0f / (float)(Nparticles)) * d_randu(seed, i);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  if (local_id == 0)
    u1 = u[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (i < Nparticles)
    u[i] = u1 + (float)(i) / (float)(Nparticles);
}

// Systematic resampling: xj[i]/yj[i] <- the particle whose CDF bracket holds u[i].
__kernel void find_index_kernel(__global const float* arrayX,
                                __global const float* arrayY,
                                __global const float* CDF,
                                __global const float* u,
                                __global float* xj,
                                __global float* yj,
                                int Nparticles) {
  const int i = get_global_id(0);
  if (i < Nparticles) {
    int index = -1;
    for (int x = 0; x < Nparticles; ++x) {
      if (CDF[x] >= u[i]) {
        index = x;
        break;
      }
    }
    if (index == -1)
      index = Nparticles - 1;
    xj[i] = arrayX[index];
    yj[i] = arrayY[index];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}