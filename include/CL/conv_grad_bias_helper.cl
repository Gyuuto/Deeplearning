
#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_grad_bias_helper ( __global float* v, __global int* v_N,
                                      __global float* delta_v, __global int* delta_N,
                                      __global int* num_unit, __global int* i )
{
    int j = get_global_id(0), k = get_global_id(1), l = get_global_id(2);

    v[l* *num_unit* *delta_N + k* *delta_N + j] = delta_v[((*i+l)* *num_unit + k)* *delta_N + j];
}
)
