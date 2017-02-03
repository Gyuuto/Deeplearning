__kernel void clMatrix_ones ( __global float* v )
{
	int gid = get_global_id(0);
	
	v[gid] = 1.0f;
}
