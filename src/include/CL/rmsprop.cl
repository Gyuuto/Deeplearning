#ifndef OCL_EXTERNAL_INCLUDE
#define OCL_EXTERNAL_INCLUDE(...) __VA_ARGS__
#endif

OCL_EXTERNAL_INCLUDE(
__kernel void rmsprop ( __global float* rmsprop_r,
						__global float* update, __global float* nabla,
						__global float* rmsprop_gamma, __global float* rmsprop_eps,
						__global float* EPS )
{
	int gid = get_global_id(0);

	rmsprop_r[gid] = *rmsprop_gamma * rmsprop_r[gid] + (1.0 - *rmsprop_gamma)*nabla[gid]*nabla[gid];
	update[gid] = -*EPS / (sqrt(rmsprop_r[gid]) + *rmsprop_eps) * nabla[gid];
}
)
