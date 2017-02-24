#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void function_Abs_diff ( __global float* y, __global float* x )
{
	int gid = get_global_id(0);

	float y_diff = 0.0;
	if( x[gid] > 1.0E-10 ) y_diff = 1.0;
	else if( x[gid] < -1.0E-10 ) y_diff = -1.0;
	
	y[gid] = y_diff;
}

)
