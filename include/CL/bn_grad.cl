__kernel void bn_grad ( __global float* tmp_nabla1, __global int* ld_na1,
						__global float* tmp_nabla2, __global int* ld_na2,
						__local float* partial_sum )
{
	int j = get_global_id(0), i = get_global_id(1);
	int lid1 = get_local_id(0), lid1_size = get_local_size(0);
	int lid2 = get_local_id(1), lid2_size = get_local_size(1);
	
	partial_sum[lid2*lid1_size+lid1] = tmp_nabla1[j* *ld_na1 + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	int offset1 = 1, offset2 = 2;
	for( int l = lid1_size; l > 0; l >>= 1 ){
		if( lid1%offset2 == 0 && lid1+offset1 < lid1_size ){
			partial_sum[lid2*lid1_size+lid1] += partial_sum[lid2*lid1_size+lid1+offset1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}
	if( lid1 == 0 ) tmp_nabla1[get_group_id(0)* *ld_na1 + i] = partial_sum[lid2*lid1_size];

	partial_sum[lid2*lid1_size+lid1] = tmp_nabla2[j* *ld_na2 + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	offset1 = 1; offset2 = 2;
	for( int l = lid1_size; l > 0; l >>= 1 ){
		if( lid1%offset2 == 0 && lid1+offset1 < lid1_size ){
			partial_sum[lid2*lid1_size+lid1] += partial_sum[lid2*lid1_size+lid1+offset1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}
	if( lid1 == 0 ) tmp_nabla2[get_group_id(0)* *ld_na2 + i] = partial_sum[lid2*lid1_size];
}

