#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_Sigmoid ( __global float* y, __global float* x, __constant float* alpha )
{
	int gid = get_global_id(0);

	y[gid] = 1.0 / (1.0 + exp(-*alpha*x[gid]));
}

)
