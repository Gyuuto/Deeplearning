__kernel void clMatrix_eye( __global float* v, __constant int* m, __constant int* n )
{
	int gid1 = get_global_id(0);

	v[gid1**n + gid1] = 1.0f;
}
