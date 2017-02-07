__kernel void conv_delta_img_set ( __global float* img, __global int* ld_img,
								   __global float* delta, __global int* nrows, __global int* ldu,
								   __global int* i, __global int* mn, __global int* l_idx,
								   __global int* delta_idx )
{
	int j = get_global_id(2), l = get_global_id(1), kst = get_global_id(0);

	int idx = j * *mn + kst % *mn, idx2 = kst / *mn;
	if( delta_idx[idx] != -1 ){
		img[(delta_idx[idx] + l * get_global_size(2)) * *ld_img + kst] =
			delta[(idx2 * *nrows + j) * *ldu + l + *i];
	}
}
