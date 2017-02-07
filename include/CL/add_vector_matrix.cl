__kernel void add_vector_matrix ( __global float* mat, __global float* vec, __global int* n, __global int* offset )
{
	int ld = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	mat[gid1*ld + gid2] += vec[gid1**n + *offset];
}
