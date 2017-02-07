__kernel void conv_apply_img_set ( __global float* img, __global int* ld_img,
								   __global float* U, __global int* nrows, __global int* ldu,
								   __global int* i, __global int* mn,
								   __global int* feed_idx )
{
	int j = get_global_id(1), l = get_global_id(2), kst = get_global_id(0);

	int idx = j * *mn + kst % *mn, idx2 = kst / *mn;
	if( feed_idx[idx] != -1 ){
		img[(l * get_global_size(1) + j)* *ld_img + kst] = U[(idx2 * *nrows + feed_idx[idx]) * *ldu + *i + l];
	}
	else{
		img[(l * get_global_size(1) + j)* *ld_img + kst] = 0.0;
	}
}
