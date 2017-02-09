__kernel void function_CrossEntropy ( __global float* y, __global float* x, __global float* d )
{
	int gid = get_global_id(0);
	
	y[gid] = d[gid]*log(x[gid]);
}

