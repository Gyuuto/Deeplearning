#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void mult_vector_matrix ( __global float* mat, __global float* vec, __global int* n )
{
	int ld = get_global_size(1), ld2 = get_global_size(0);
	int gid1 = get_global_id(0), gid2 = get_global_id(1), gid3 = get_global_id(2);

	mat[gid3*ld*ld2 + gid1*ld + gid2] *= vec[gid1**n + gid3];
}
)
