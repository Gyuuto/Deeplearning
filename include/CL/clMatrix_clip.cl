#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void clMatrix_clip( __global float* x, __global float* val )
{
	int gid1 = get_global_id(0);

	if( x[gid1] > *val ) x[gid1] = *val;
	else if( x[gid1] < -*val ) x[gid1] = -*val;
}
)
