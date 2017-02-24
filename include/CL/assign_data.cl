#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void assign_data ( __global float* y, __global float* x, __global int* idx, __global int* offset, __constant int* N )
{
	int ld = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	int i = gid2 + *offset;
	if( i >= *N ) i -= *N;
	
	y[gid1*ld + gid2] = x[gid1**N + idx[i]];
}
)
