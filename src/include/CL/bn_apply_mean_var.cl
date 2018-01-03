#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_apply_mean_var ( __global float* mean, __global int* ld_mean,
								  __global float* var, __global int* ld_var,
								  __global float* U, __global int* prev_num_unit, __global int* ld_U )
{
	int j = get_global_id(0), i = get_global_id(1);

    float c = 0.0;
	mean[i* *ld_mean + j] = 0.0;
	for( int k = 0; k < *ld_U; ++k ){
        float y = U[(i* *prev_num_unit + j)* *ld_U + k] - c;
        float t = mean[i* *ld_mean + j] + y;
        c = (t - mean[i* *ld_mean + j]) - y;
		mean[i* *ld_mean + j] = t;
	}
	mean[i* *ld_mean + j] /= *ld_U;
	
    c = 0.0;
    var[i* *ld_var + j] = 0.0;
	for( int k = 0; k < *ld_U; ++k ){
		float tmp = (U[(i* *prev_num_unit + j)* *ld_U + k] - mean[i* *ld_mean + j]);
        float y = tmp*tmp - c;
        float t = var[i* *ld_var + j] + y;
        c = (t - var[i* *ld_var + j]) - y;
		var[i* *ld_var + j] = t;
	}
	var[i* *ld_var + j] /= *ld_U;
}
)
