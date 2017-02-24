#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_grad_helper ( __global float* tmp_nabla1, __global int* ld_na1,
							   __global float* tmp_nabla2, __global int* ld_na2,
							   __global float* delta, __global float* U_apply,
							   __global int* num_unit, __global int* ld_U,
							   __global float* mean, __global int* ld_mean,
							   __global float* var, __global int* ld_var,
							   __global float* EPS )
{
	int j = get_global_id(0) / *ld_U, i = get_global_id(1), k = get_global_id(0) % *ld_U;

	tmp_nabla1[i* *ld_na1 + (k* *num_unit + j)] =
		delta[(i* *num_unit + j)* *ld_U + k]*(U_apply[(i* *num_unit + j)* *ld_U + k] - mean[i* *ld_mean + j])/sqrt(var[i* *ld_var + j] + *EPS);

	tmp_nabla2[i* *ld_na1 + k* *num_unit + j] = delta[(i* *num_unit + j)* *ld_U + k];
}
)
