#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_grad_delta_set ( __global float* delta_mat, __global int* ld_delta,
									__global float* delta, __global int* nrows, __global int* ldu,
									__global int* i )
{
	int size = get_global_size(1), my_size = get_global_size(2), num_map = get_global_size(0);
	int l = get_global_id(1), k = get_global_id(2), j = get_global_id(0);

	delta_mat[(l*my_size + k)* *ld_delta + j] = delta[(j* *nrows + k)* *ldu + l + *i];
}
)