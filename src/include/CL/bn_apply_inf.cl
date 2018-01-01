#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void bn_apply_inf ( __global float* ret, __global int* num_unit, __global int* ld_ret,
                             __global float* mean, __global int* ld_mean,
                             __global float* var, __global int* ld_var,
                             __global float* W, __global float* b, __global float* EPS,
                             __global float* U, __global int* prev_num_unit, __global int* ld_U )
{
    
}
)
