__kernel void clMatrix_sub ( __global float* out, __global float* in,
							 __constant int* sx, __constant int* sy, __constant int* ld_in )
{
	int gid2_size = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	out[gid1*gid2_size + gid2] = in[(*sy + gid1)**ld_in + *sx+gid2];
}
