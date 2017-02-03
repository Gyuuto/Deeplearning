__kernel void function_TruncatedPower_diff ( __global float* y, __global float* x, __constant int* n )
{
	int gid = get_global_id(0);

	y[gid] = (x[gid] < 0.0 ? 0.0 : *n*pown(x[gid], *n-1));
}

