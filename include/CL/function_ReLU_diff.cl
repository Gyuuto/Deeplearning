__kernel void function_ReLU_diff ( __global float* y, __global float* x )
{
	int gid = get_global_id(0);

	y[gid] = (x[gid] <= 0.0 ? 0.0 : 1.0);
}

