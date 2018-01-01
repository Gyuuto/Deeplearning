#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void clMatrix_add( __global float* out, __global float* alpha, __global float* x, __global float* y )
{
	int gid1 = get_global_id(0);

	out[gid1] = *alpha * x[gid1] + y[gid1];
}
)
