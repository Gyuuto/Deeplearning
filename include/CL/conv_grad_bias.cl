#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_grad_bias ( __global float* v, __global int* v_N, __local float* partial_sum )        
{
    int i = get_global_id(0), j = get_global_id(1);
    int once_map = get_global_size(1);
    int lc1 = get_local_id(0), lc1_size = get_local_size(0);
    int lc2 = get_local_id(1), lc2_size = get_local_size(1);

    partial_sum[lc2*lc1_size + lc1] = v[j*once_map + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    int offset1 = 1, offset2 = 2;
    float c = 0.0;
    for( int k = lc1_size; k > 0; k >>= 1 ) {
        if( lc1%offset2 == 0 && lc1+offset1 < lc1_size ) {
            float y = partial_sum[lc2*lc1_size + lc1 + offset1] - c;
            float t = partial_sum[lc2*lc1_size + lc1] + y;
            c = (t - partial_sum[lc2*lc1 + lc1]) - y;

            partial_sum[lc2*lc1_size + lc1] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        offset1 <<= 1; offset2 <<= 1;
    }
    if( lc1 == 0 ) v[j*once_map + get_group_id(0)] = partial_sum[lc2*lc1_size];
}
)
