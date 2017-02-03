__kernel void function_Polynomial_diff ( __global float* y, __global float* x, __constant int* n )
{
	int gid = get_global_id(0);

	y[gid] = *n*pown(x[gid], *n-1);
}

