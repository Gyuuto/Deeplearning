#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void maxpool_apply ( __global float* ret, __global int* num_unit, __global int* ld_d,
							  __global float* U, __global float* S,
							  __global int* prev_num_unit, __global int* ld_u,
							  __global int* prev_ldu, __global int* ldu,
							  __global int* stride, __global int* pad,
							  __global int* m, __global int* n )
{
	int j = get_global_id(0), i = get_global_id(2), k = get_global_id(1);

	int x = j % *ldu, y = j / *ldu;

	int prev_width = *prev_ldu, prev_height = *prev_num_unit / *prev_ldu;
	int gap = prev_width + 2* *pad;
	int max_idx = -1;
	float max_val;

	for( int s = 0; s < *m; ++s ){
		for( int t = 0; t < *n; ++t ){
			int idx = *stride*x + t + s*gap + *stride*y*gap;
			int nx = idx%gap - *pad, ny = idx/gap - *pad;

			if( nx < 0 || nx >= prev_width || ny < 0 || ny >= prev_height ) continue;

			int idx_ = ny* *prev_ldu + nx;
			if( max_idx == -1 || max_val < U[(i* *prev_num_unit + idx_)* *ld_u + k] ){
				max_val = U[(i* *prev_num_unit + idx_)* *ld_u + k];
				max_idx = idx_;
			}
		}
	}
	
	ret[(i* *num_unit + j)* *ld_d + k] = U[(i* *prev_num_unit + max_idx)* *ld_u + k];
	S[(i* *num_unit + j)* *ld_d + k] = max_idx;
}
)