__kernel void function_Polynomial ( __global float* y, __global float* x, __constant int* n )
{
	int gid = get_global_id(0);

	y[gid] = pown(x[gid], *n);
}

