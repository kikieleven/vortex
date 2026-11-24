// Parboil spmv (src/opencl_base/kernel.cl) — sparse matrix-vector product in
// padded JDS format.
//
// Ported faithfully: the indexing, the per-warp-group diagonal bound
// (sh_zcnt_int[ix/32]) and the JDS data layout are unchanged. The upstream
// host chooses the launch geometry from device properties; here the kernel is
// launched 1D with 16-work-item groups (Vortex's work-group cap). The kernel
// already indexes by global id only, so nothing else changes.
//
// Layout produced by the host (see main.cc):
//   row i of JDS (i in [0,dim)) is original row perm[i];
//   diagonal k occupies [k*dim, (k+1)*dim);
//   jds_ptr_int[k] = k*dim, sh_zcnt_int[g] = number of diagonals for group g.

__kernel void spmv_jds_naive(__global float *dst_vector, __global float *d_data,
			     __global int *d_index, __global int *d_perm,
			     __global float *x_vec, const int dim,
			     __constant int *jds_ptr_int,
			     __constant int *sh_zcnt_int)
{
	int ix = get_global_id(0);

	if (ix < dim) {
		float sum = 0.0f;
		// 32 is warp size
		int bound=sh_zcnt_int[ix/32];

		for(int k=0;k<bound;k++)
		{
			int j = jds_ptr_int[k] + ix;
			int in = d_index[j];

			float d = d_data[j];
			float t = x_vec[in];

			sum += d*t;
		}

		dst_vector[d_perm[ix]] = sum;
	}
}