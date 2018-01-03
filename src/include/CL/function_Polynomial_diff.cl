#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_Polynomial_diff ( __global float* x, __global int* n )
{
	int gid = get_global_id(0);

	x[gid] = *n*pown(x[gid], *n-1);
}

)