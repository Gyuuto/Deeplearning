#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_Sigmoid ( __global float* x, __global float* alpha )
{
	int gid = get_global_id(0);

	x[gid] = 1.0 / (1.0 + exp(-*alpha*x[gid]));
}

)
