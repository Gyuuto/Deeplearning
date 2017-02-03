__kernel void full_delta_init ( __global float* v, __global float* u )
{
	int ld = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	v[gid1*ld + gid2] = u[(gid1+1)*ld + gid2];
}
