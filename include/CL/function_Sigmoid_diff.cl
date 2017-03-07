#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_Sigmoid_diff ( __global float* y, __global float* x, __global float* alpha )
{
	int gid = get_global_id(0);

	float tmp = 1.0 + exp(-*alpha*x[gid]);
	y[gid] = *alpha*exp(-*alpha*x[gid])/(tmp*tmp);
}

)
