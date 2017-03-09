#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_delta ( __global float* nx_delta, __global int* prev_num_unit, __global int* ld_nx,
						 __global float* U_apply, __global float* U_diff, __global float* delta, __global int* ld_U,
						 __global float* mean, __global int* ld_mean,
						 __global float* var, __global int* ld_var,
						 __global float* W, __global float* EPS )
{
	int k = get_global_id(0), j = get_global_id(1), i = get_global_id(2);

	float tmp1 = 0.0f, tmp2 = 0.0f;
	for( int l = 0; l < *ld_U; ++l ){
		tmp1 += delta[(i* *prev_num_unit + j)* *ld_U + l];
		tmp2 += delta[(i* *prev_num_unit + j)* *ld_U + l] * (U_apply[(i* *prev_num_unit + j)* *ld_U + l] - mean[i* *ld_mean + j]);
	}
	tmp1 /= *ld_U;
	tmp2 /= *ld_U;

	nx_delta[(i* *prev_num_unit + j)* *ld_U + k] = 
		W[i]/sqrt(var[i* *ld_var + j] + *EPS)*delta[(i* *prev_num_unit + j)* *ld_U + k]*U_diff[(i* *prev_num_unit + j)* *ld_U + k] -
		W[i]/sqrt(var[i* *ld_var + j] + *EPS)*U_diff[(i* *prev_num_unit + j)* *ld_U + k]*tmp1 -
		W[i]/(powr(var[i* *ld_var + j], 1.5f) + *EPS)*U_diff[(i* *prev_num_unit + j)* *ld_U + k]*(U_apply[(i* *prev_num_unit + j)* *ld_U + k] - mean[i* *ld_mean + j])*tmp2;
}
)
