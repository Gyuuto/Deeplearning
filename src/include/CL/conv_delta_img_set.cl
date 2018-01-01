#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_delta_img_set ( __global float* img, __global int* ld_img,
								   __global float* delta, __global int* nrows, __global int* ldu,
								   __global int* i, __global int* mn, __global int* my_size,
								   __global int* delta_idx )
{
	int j = get_global_id(2), l = get_global_id(1), kst = get_global_id(0);

	int idx = j * *mn + (kst % *mn), idx2 = kst / *mn;
	if( delta_idx[idx] != -1 ){
		img[(delta_idx[idx] + l * *my_size) * *ld_img + kst] = 
			delta[(idx2 * *nrows + j) * *ldu + l + *i];
	}
}
)
