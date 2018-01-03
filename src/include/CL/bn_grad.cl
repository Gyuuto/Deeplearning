#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_grad ( __global float* tmp_nabla1, __global int* ld_na1,
						__global float* tmp_nabla2, __global int* ld_na2,
						__local float* partial_sum )
{
	int j = get_global_id(0), i = get_global_id(1);
	int lid1 = get_local_id(0), lid1_size = get_local_size(0);
	int lid2 = get_local_id(1), lid2_size = get_local_size(1);

	partial_sum[lid2*lid1_size+lid1] = tmp_nabla1[i* *ld_na1 + j];
	barrier(CLK_LOCAL_MEM_FENCE);

	int offset1 = 1, offset2 = 2;
	float c = 0.0;
	for( int l = lid1_size; l > 0; l >>= 1 ){
		if( lid1%offset2 == 0 && lid1+offset1 < lid1_size ){
			float y = partial_sum[lid2*lid1_size+lid1+offset1] - c;
			float t = partial_sum[lid2*lid1_size+lid1] + y;
			c = (t - partial_sum[lid2*lid1_size+lid1]) - y;

			partial_sum[lid2*lid1_size+lid1] = t;
			/* partial_sum[lid2*lid1_size+lid1] += partial_sum[lid2*lid1_size+lid1+offset1]; */
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}
	if( lid1 == 0 ) tmp_nabla1[i* *ld_na1 + get_group_id(0)] = partial_sum[lid2*lid1_size];

	partial_sum[lid2*lid1_size+lid1] = tmp_nabla2[i* *ld_na2 + j];
	barrier(CLK_LOCAL_MEM_FENCE);

	offset1 = 1; offset2 = 2;
    c = 0.0;
	for( int l = lid1_size; l > 0; l >>= 1 ){
		if( lid1%offset2 == 0 && lid1+offset1 < lid1_size ){
            float y = partial_sum[lid2*lid1_size+lid1+offset1] - c;
            float t = partial_sum[lid2*lid1_size+lid1] + y;
            c = (t - partial_sum[lid2*lid1_size+lid1]) - y;

            partial_sum[lid2*lid1_size+lid1] = t;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		offset1 <<= 1; offset2 <<= 1;
	}
	if( lid1 == 0 ) tmp_nabla2[i* *ld_na2 + get_group_id(0)] = partial_sum[lid2*lid1_size];
}

)
