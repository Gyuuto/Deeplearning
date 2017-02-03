__kernel void function_Softmax ( __global float* y, __global float* x, __global float* max_x, __global float* sum_x )
{
	int ld = get_global_size(1);
	int gid1 = get_global_id(0), gid2 = get_global_id(1);

	y[gid1*ld + gid2] = exp(x[gid1*ld + gid2] - max_x[gid2]) / sum_x[gid2];
}

