// Parboil stencil (src/opencl_base/kernel.cl) — seven-point 3D stencil,
// Anext[i][j][k] = c1 * (six neighbours) - c0 * A[i][j][k].
//
// Ported faithfully: coefficients, neighbour order, Index3D layout and the
// "write interior only" semantics are unchanged. The upstream kernel already
// reads only global ids, so the only adaptation is the host launch: Vortex
// caps a work-group at 16, so the x dimension uses a 16-work-item group and
// the grid is rounded up; the existing `if (i < nx-1)` guard skips the extra
// work-items. The j/k grid dimensions are ny-2 / nz-2 (as upstream), so they
// need no guard.

#include "common.h"

__kernel void naive_kernel(float c0,float c1,__global float* A0,__global float *Anext,int nx,int ny,int nz)
{
	int i = get_global_id(0)+1;
	int j = get_global_id(1)+1;
	int k = get_global_id(2)+1;

	if(i<nx-1)
	{
		Anext[Index3D (nx, ny, i, j, k)] = c1 *
		( A0[Index3D (nx, ny, i, j, k + 1)] +
		  A0[Index3D (nx, ny, i, j, k - 1)] +
		  A0[Index3D (nx, ny, i, j + 1, k)] +
		  A0[Index3D (nx, ny, i, j - 1, k)] +
		  A0[Index3D (nx, ny, i + 1, j, k)] +
		  A0[Index3D (nx, ny, i - 1, j, k)] )
		- A0[Index3D (nx, ny, i, j, k)] * c0;
	}
}