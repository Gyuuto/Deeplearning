#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_CrossEntropy ( __global float* x, __global float* d )
{
	int gid = get_global_id(0);
	
	x[gid] = d[gid]*log(x[gid]);
}

)
