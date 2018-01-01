#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void clMatrix_eye( __global float* v, __global int* m, __global int* n )
{
	int gid1 = get_global_id(0);

	v[gid1**n + gid1] = 1.0f;
}
)
