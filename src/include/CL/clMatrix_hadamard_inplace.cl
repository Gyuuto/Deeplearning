#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void clMatrix_hadamard_inplace ( __global float* out, __global float* in )
{
	int gid = get_global_id(0);
	
	out[gid] *= in[gid];
}
)
