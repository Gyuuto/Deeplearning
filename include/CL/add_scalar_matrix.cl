__kernel void add_scalar_matrix ( __global float* mat, __global float* scalar )
{
	int gid = get_global_id(0);

	mat[gid] += *scalar;
}
