#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void full_apply_init ( __global float* v, __global float* u, __constant bool* use_bias )
{
	int ld = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	if( gid1 == 0 ){
		if( *use_bias ) v[gid1*ld + gid2] = 1.0;
		else v[gid1*ld + gid2] = 0.0;
	}
	else v[gid1*ld + gid2] = u[(gid1-1)*ld + gid2];
}
)
