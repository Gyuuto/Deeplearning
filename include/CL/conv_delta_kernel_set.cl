#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void conv_delta_kernel_set ( __global float* ker, __global float* W )
{
	int num_map = get_global_size(1), prev_num_map = get_global_size(0), mn = get_global_size(2);
	int gid1 = get_global_id(1), gid2 = get_global_id(0), gid3 = get_global_id(2);

	ker[(gid1*mn + gid3)*prev_num_map + gid2] = W[(gid2*mn + gid3)*num_map + gid1];
}
)
