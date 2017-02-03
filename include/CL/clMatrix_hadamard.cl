__kernel void clMatrix_hadamard ( __global float* out, __global float* in1, __global float* in2 )
{
	int gid = get_global_id(0);
	
	out[gid] = in1[gid] * in2[gid];
}
