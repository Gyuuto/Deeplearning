#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_apply_add_bias ( __global float* mat, __global float* bias )
{
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	mat[gid2*get_global_size(0) + gid1] += bias[gid2];
}
)
