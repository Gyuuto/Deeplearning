__kernel void clMatrix_sub_in ( __global float* y, __global float* x, __global int* ld_y, __global int* sx, __global int* sy )
{
	int ld_x = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	y[(*sy + gid1) * *ld_y + gid2 + *sx] = x[gid1 * ld_x + gid2];
}
