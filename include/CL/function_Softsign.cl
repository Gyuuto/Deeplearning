__kernel void function_Softsign ( __global float* y, __global float* x )
{
	int gid = get_global_id(0);

	y[gid] = x[gid] / (1.0 + fabs(x[gid]));
}

