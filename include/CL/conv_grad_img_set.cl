#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_grad_img_set ( __global float* U_mat, __global int* ld_Umat,
								   __global float* U_, __global int* nrows, __global int* ldu,
								   __global int* i, __global int* mn, __global int* l_idx,
								   __global int* delta_idx )
{
	int j = get_global_id(1) + *l_idx, l = get_global_id(0), kst = get_global_id(2);
	int prev_num_map = get_global_size(2) / *mn;
	int my_size = get_global_size(1);

	int idx = j * *mn + kst % *mn, k = kst / *mn;
	if( delta_idx[idx] != -1 ){
		U_mat[kst* *ld_Umat + (j - *l_idx) + l*my_size] =
			U_[(k* *nrows + delta_idx[idx])* *ldu + l + *i];
	}
	else{
		U_mat[kst* *ld_Umat + (j - *l_idx) + l*my_size] = 0.0;
	}
}
)
