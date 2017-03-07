#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
	__kernel void function_Pow ( __global float* y, __global float* x, __global float* n )
{
	int gid = get_global_id(0);

	y[gid] = pow(x[gid], *n);
}

)
