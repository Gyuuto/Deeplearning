__kernel void bn_grad_helper ( __global float* y, __global int* ld_y,
							   __global float* x, __global int* ld_x,
							   __local float* partial_sum )
{
	int gid1 = get_global_id(0), gid2 = get_global_id(1);
	int lid = get_local_id(0), lid_size = get_local_size(0);

	partial_sum[lid] = x[gid2* *ld_y + gid1];
	barrier(CLK_LOCAL_MEM_FENCE);

	int offset1 = 1, offset2 = 2;
	for( int i = lid_size; i > 0; i >>= 1 ){
		if( lid%offset2 == 0 && lid+offset1 < lid_size ){
			partial_sum[lid] += partial_sum[lid+offset1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}

	if( lid == 0 ) y[gid2* *ld_y + get_group_id(0)] = partial_sum[0];
}
