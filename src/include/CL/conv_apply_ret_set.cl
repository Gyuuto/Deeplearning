#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_apply_ret_set ( __global float* ret, __global int* ld_ret,
								   __global float* tmp_img, __global int* ld_tmp_img, 
								   __global int* offset, __global int* my_size )
{
	int k = get_global_id(1), i = get_global_id(0), j = get_global_id(2);

	ret[(j*get_global_size(1) + k) * *ld_ret + *offset + i] = tmp_img[(k + i * *my_size) * *ld_tmp_img + j];
}
)