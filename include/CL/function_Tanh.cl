__kernel void function_Tanh ( __global float* y, __global float* x )
{
	int gid = get_global_id(0);

	y[gid] = tanh(x[gid]);
}

