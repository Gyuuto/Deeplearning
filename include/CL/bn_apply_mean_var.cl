#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_apply_mean_var ( __global float* mean, __global int* ld_mean,
								  __global float* var, __global int* ld_var,
								  __global float* U, __global int* prev_num_unit, __global int* ld_U )
{
	int j = get_global_id(0), i = get_global_id(1);

	mean[i* *ld_mean + j] = 0.0;
	for( int k = 0; k < *ld_U; ++k ){
		mean[i* *ld_mean + j] += U[(i* *prev_num_unit + j)* *ld_U + k];
	}
	mean[i* *ld_mean + j] /= *ld_U;
	
	var[i* *ld_mean + j] = 0.0;
	for( int k = 0; k < *ld_U; ++k ){
		float tmp = (U[(i* *prev_num_unit + j)* *ld_U + k] - mean[i* *ld_var + j]);
		var[i* *ld_var + j] += tmp*tmp;
	}
	var[i* *ld_mean + j] /= *ld_U;
}
)
