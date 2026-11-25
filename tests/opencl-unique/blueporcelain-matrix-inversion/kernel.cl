// Blue Porcelain basic/MATRIX_INVERSION — CUDA to OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/MATRIX_INVERSION/matrix_inversion.cu
// Same four-step Gauss-Jordan update, applied once per pivot i:
//   nodiag_normalize : row i, columns != i  -> divide A and I by A[i][i]
//   diag_normalize   : the pivot element    -> divide A and I by A[i][i]
//   gaussjordan      : rows != i            -> eliminate column i from A and I
//   set_zero         : A[x][i] for x != i   -> 0
//
// Adaptations for Vortex:
//   - 4x4 work-groups (16 work-items) instead of 8x8; global size is rounded up
//     and the kernels keep upstream's `x < n && y < n` guards.
//   - x is the row (global dim 0) and y the column (global dim 1), matching
//     upstream's `x = blockIdx.x * blockDim.x + threadIdx.x`.

__kernel void nodiag_normalize(__global float *A,
                               __global float *I,
                               int n,
                               int i)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < n && y < n)
    if (x == i && x != y) {
      float p = A[i * n + i];
      I[x * n + y] /= p;
      A[x * n + y] /= p;
    }
}

__kernel void diag_normalize(__global float *A,
                             __global float *I,
                             int n,
                             int i)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < n && y < n)
    if (x == y && x == i) {
      float p = A[i * n + i];
      I[x * n + y] /= p;
      A[x * n + y] /= p;
    }
}

__kernel void gaussjordan(__global float *A,
                          __global float *I,
                          int n,
                          int i)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < n && y < n)
    if (x != i) {
      float f = A[x * n + i];
      I[x * n + y] -= I[i * n + y] * f;
      if (y != i)
        A[x * n + y] -= A[i * n + y] * f;
    }
}

__kernel void set_zero(__global float *A,
                       __global float *I,
                       int n,
                       int i)
{
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < n && y < n)
    if (x != i && y == i)
      A[x * n + y] = 0.0f;
}