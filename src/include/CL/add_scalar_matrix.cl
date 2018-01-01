#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void add_scalar_matrix ( __global float* mat, __global float* scalar )
{
	int gid = get_global_id(0);

	mat[gid] += *scalar;
}
)
