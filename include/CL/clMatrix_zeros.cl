__kernel void clMatrix_zeros ( __global float* v )
{
	int gid = get_global_id(0);

	v[gid] = 0.0f;
}
