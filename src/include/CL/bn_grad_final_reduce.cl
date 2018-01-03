#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_grad_final_reduce ( __global float* tmp_nabla1, __global int* ld_na1,
									 __global float* tmp_nabla2, __global int* ld_na2,
									 __global int* n )
{
	int j = get_global_id(0);

	for ( int i = 1; i < *n; ++i ){
		tmp_nabla1[j* *ld_na1] += tmp_nabla1[j* *ld_na1 + i];
		tmp_nabla2[j* *ld_na2] += tmp_nabla2[j* *ld_na2 + i];
	}
}
									 
)