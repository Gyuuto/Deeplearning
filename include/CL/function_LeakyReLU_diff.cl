#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_LeakyReLU_diff ( __global float* y, __global float* x, __global float* alpha )
{
	int gid = get_global_id(0);

	y[gid] = (x[gid] <= 0.0 ? *alpha : 1.0f);
}

)
