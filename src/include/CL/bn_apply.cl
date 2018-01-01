#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_apply ( __global float* ret, __global int* num_unit, __global int* ld_ret,
						 __global float* mean, __global int* ld_mean,
						 __global float* var, __global int* ld_var,
						 __global float* W, __global float* b, __global float* EPS,
						 __global float* U, __global int* prev_num_unit, __global int* ld_U )
{
	int k = get_global_id(0), j = get_global_id(1), i = get_global_id(2);

	ret[(i* *num_unit + j)* *ld_ret + k] = W[i]*(U[(i* *prev_num_unit + j)* *ld_U + k] - mean[i* *ld_mean + j]) / sqrt(var[i* *ld_var + j] + *EPS) + b[i];
}
)
