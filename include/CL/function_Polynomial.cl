#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_Polynomial ( __global float* y, __global float* x, __global int* n )
{
	int gid = get_global_id(0);

	y[gid] = pown(x[gid], *n);
}

)
