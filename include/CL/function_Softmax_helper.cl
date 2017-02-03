__kernel void function_Softmax_helper ( __global float* x, __global float* max_x, __global float* sum_x, __constant int* m, __constant int* n )
{
	int i;
	int gid = get_global_id(0);

	max_x[gid] = x[gid**n];
	for( i = 0; i < *m; ++i ){
		if( max_x[gid] < x[i**n + gid] ){
			max_x[gid] = x[i**n + gid];
		}
	}

	sum_x[gid] = 0.0f;
	for( i = 0; i < *m; ++i ){
		sum_x[gid] += exp(x[i**n + gid] - max_x[gid]);
	}
}
