#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void add_L2_regular ( __global float* nabla_w, __global float* w, __constant float* lambda )
{
	int ld = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	nabla_w[gid1*ld + gid2+1] += *lambda * w[gid1*ld + gid2+1];
}
)
