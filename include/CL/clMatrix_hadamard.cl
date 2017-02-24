#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void clMatrix_hadamard ( __global float* out, __global float* in1, __global float* in2 )
{
	int gid = get_global_id(0);
	
	out[gid] = in1[gid] * in2[gid];
}
)
