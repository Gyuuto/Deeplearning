__kernel void function_Sigmoid_diff ( __global float* y, __global float* x, __constant float* alpha )
{
	int gid = get_global_id(0);

	float tmp = 1.0 + exp(-*alpha*x[gid]);
	y[gid] = *alpha*exp(-*alpha*x[gid])/(tmp*tmp);
}

