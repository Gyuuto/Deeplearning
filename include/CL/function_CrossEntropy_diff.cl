__kernel void function_ReLU ( __global float* y, __global float* x )
{
	int gid = get_global_id(0);

	y[gid] = (x[gid] <= 0.0 ? 0.0 : x[gid]);
}

