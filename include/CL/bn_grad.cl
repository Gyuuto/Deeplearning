__kernel void bn_grad ( __global float* tmp_nabla1, __global int* ld_na1,
						__global float* tmp_nabla2, __global int* ld_na2,
						__global float* delta, __global float* U_apply,
						__global int* num_unit, __global int* ld_U,
						__global float* mean, __global int* ld_mean,
						__global float* var, __global int* ld_var,
						__global float* EPS, __local float* partial_sum )
{
	int j = get_global_id(0) / *ld_U, i = get_global_id(1), k = get_global_id(0) % *ld_U;
	int lid = get_local_id(0), lid_size = get_local_size(0);
	
	partial_sum[lid] = delta[(i* *num_unit + j)* *ld_U + k]*(U_apply[(i* *num_unit + j)* *ld_U + k] - mean[i* *ld_mean + j])/sqrt(var[i* *ld_var + j] + *EPS);
	barrier(CLK_LOCAL_MEM_FENCE);

	int offset1 = 1, offset2 = 2;
	for( int l = lid_size; l > 0; l >>= 1 ){
		if( lid%offset2 == 0 && lid+offset1 < lid_size ){
			partial_sum[lid] += partial_sum[lid+offset1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}
	if( lid == 0 ) tmp_nabla1[i* *ld_na1 + get_group_id(0)] = partial_sum[0];

	partial_sum[lid] = delta[(i* *num_unit + j)* *ld_U + k];
	barrier(CLK_LOCAL_MEM_FENCE);

	offset1 = 1; offset2 = 2;
	for( int l = lid_size; l > 0; l >>= 1 ){
		if( lid%offset2 == 0 && lid+offset1 < lid_size ){
			partial_sum[lid] += partial_sum[lid+offset1];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}
	if( lid == 0 ) tmp_nabla2[i* *ld_na2 + get_group_id(0)] = partial_sum[0];
}

