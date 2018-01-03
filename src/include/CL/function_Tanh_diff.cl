#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_Tanh_diff ( __global float* x )
{
	int gid = get_global_id(0);

	float tmp = tanh(x[gid]);
	x[gid] = 1.0 - tmp*tmp;
}

)