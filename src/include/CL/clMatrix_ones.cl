#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void clMatrix_ones ( __global float* v )
{
	int gid = get_global_id(0);
	
	v[gid] = 1.0f;
}
)
