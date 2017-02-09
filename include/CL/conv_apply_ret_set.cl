__kernel void conv_apply_ret_set ( __global float* ret, __global int* ld_ret,
								   __global float* tmp_img, __global int* ld_tmp_img, 
								   __global int* offset, __global int* my_size )
{
	int k = get_global_id(2), i = get_global_id(0), j = get_global_id(1);

	if( *offset*get_global_size(0) + i < *ld_ret )
		ret[(j*get_global_size(2) + k) * *ld_ret + *offset*get_global_size(0) + i] = tmp_img[(k + i * *my_size) * *ld_tmp_img + j];
}
