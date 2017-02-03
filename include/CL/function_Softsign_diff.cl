__kernel void function_Softsign_diff ( __global float* y, __global float* x )
{
	int gid = get_global_id(0);

	float tmp = 1.0 + fabs(x[gid]);
	float y_diff = 0.0;

	if( x[gid] > 1.0E-10 ) y_diff = 1.0;
	else if( x[gid] < 1.0E-10 ) y_diff = -1.0;
	
	y[gid] = (tmp - x[gid]*y_diff) / (tmp*tmp);
}

