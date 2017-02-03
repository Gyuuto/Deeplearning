__kernel void adam ( __global float* adam_v, __global float* adam_r,
					 __global float* update_w, __global float* nabla_w,
					 __constant float* adam_beta, __constant float* adam_gamma,
					 __constant float* adam_beta_, __constant float* adam_gamma_,
					 __constant float* EPS, __constant float* adam_eps )
{
	int gid = get_global_id(0);

	adam_v[gid] = *adam_beta * adam_v[gid] + (1.0 - *adam_beta)*nabla_w[gid];
	adam_r[gid] = *adam_gamma * adam_r[gid] + (1.0 - *adam_gamma)*nabla_w[gid]*nabla_w[gid];

	float v_hat = adam_v[gid] / (1.0 - *adam_beta_);
	float r_hat = adam_r[gid] / (1.0 - *adam_gamma_);
	update_w[gid] = -*EPS * v_hat / (sqrt(r_hat) + *adam_eps);
}
