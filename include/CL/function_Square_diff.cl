__kernel void function_Square_diff ( __global float* y, __global float* x, __global float* d )
{
	int gid = get_global_id(0);

	y[gid] = 2.0f*(x[gid] - d[gid]);
}

