// Blue Porcelain basic/TRIANGLE_AREA — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/TRIANGLE_AREA/triangle_area.cu
// The kernel body is unchanged: read the three sides, apply the triangle
// inequality, write Heron's area or 0. Adaptation: 1D work-groups of 16 and an
// `idx < n` guard (upstream has no guard).

__kernel void gpu_triangle_area(__global const float *arr1,
                                __global const float *arr2,
                                __global const float *arr3,
                                __global float *res,
                                int n)
{
  int idx = get_global_id(0);
  if (idx >= n)
    return;

  float a = arr1[idx];
  float b = arr2[idx];
  float c = arr3[idx];

  if (a + b > c && a + c > b && b + c > a) {
    float p = (a + b + c) * 0.5f;
    float area = sqrt(p * (p - a) * (p - b) * (p - c));
    res[idx] = area;
  } else {
    res[idx] = 0.0f;
  }
}