#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void averagepool_delta ( __global float* nx_delta, __global int* nrows_d, __global int* ld_d,
								  __global float* U_diff, __global float* delta,
								  __global int* nrows_u, __global int* ld_u,
								  __global int* prev_ldu, __global int* ldu,
								  __global int* stride, __global int* pad,
								  __global int* m, __global int* n )
{
	int j = get_global_id(0), i = get_global_id(2), k = get_global_id(1);

	int x = j % *ldu, y = j / *ldu;

	int prev_width = *prev_ldu, prev_height = *nrows_d / *prev_ldu;
	int gap = prev_width + 2* *pad;

	for( int s = 0; s < *m; ++s ){
		for( int t = 0; t < *n; ++t ){
			int idx = *stride*x + t + s*gap + *stride*y*gap;
			int nx = idx%gap - *pad, ny = idx/gap - *pad;

			if( nx < 0 || nx >= prev_width || ny < 0 || ny >= prev_height ) continue;

			int idx_ = ny* *prev_ldu + nx;
			nx_delta[(i* *nrows_d + idx_)* *ld_d + k] = delta[(i* *nrows_u + j)* *ld_u + k] * U_diff[(i* *nrows_d + idx_)* *ld_d + k] / (*m * *n);
		}
	}
}
)
