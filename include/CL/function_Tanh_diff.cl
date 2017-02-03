__kernel void function_Tanh_diff ( __global float* y, __global float* x )
{
	int gid = get_global_id(0);

	float tmp = tanh(x[gid]);
	y[gid] = 1.0 - tmp*tmp;
}

