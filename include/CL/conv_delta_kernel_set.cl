__kernel void conv_delta_kernel_set ( __global float* ker, __global float* W )
{
	int num_map = get_global_size(2), prev_num_map = get_global_size(0), mn = get_global_size(1);
	int gid1 = get_global_id(2), gid2 = get_global_id(0), gid3 = get_global_id(1);

	ker[(gid1*mn + gid3)*prev_num_map + gid2] = W[(gid2*mn + gid3)*num_map + gid1];
}
