__kernel void function_Square ( __global float* y, __global float* x, __global float* d, __local float* partial_sum )
{
	int lid = get_local_id(0);
	int lid_size = get_local_size(0);
	int offset1 = 1, offset2 = 2;

	float tmp = x[get_global_id(0)] - d[get_global_id(0)];
	partial_sum[lid] = tmp*tmp;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for( int i = lid_size; i > 0; i >>= 1 ){
		if( lid%offset2 == 0 && lid+offset1 < lid_size ){
			partial_sum[lid] += partial_sum[lid+offset1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}

	if( lid == 0 ) y[get_group_id(0)] = partial_sum[0];
}

