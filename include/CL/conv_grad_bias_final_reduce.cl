#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_grad_bias_final_reduce ( __global float* tmp_bias, __global int* ld_b,
											__global int* n )
{
	int j = get_global_id(0);

	for( int i = 1; i < *n; ++i ){
		tmp_bias[j* *ld_b] += tmp_bias[j* *ld_b + i];
	}
}
)
