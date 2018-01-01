#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void full_grad_bias ( __global float* nabla_b, __global float* delta, __global int* n )
{
	int m = get_global_id(0);

    nabla_b[m] = 0.0f;
    for( int i = 0; i < *n; ++i ) nabla_b[m] += delta[m* *n + i];
}
)
