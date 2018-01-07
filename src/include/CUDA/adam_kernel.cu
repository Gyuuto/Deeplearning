__global__
void adam ( const int m, const int n,
            float* v, float* r,
            float* update, float* nabla,
            const float beta, const float gamma,
            const float beta_, const float gamma_,
            const float EPS, const float adam_eps ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  
  if( i < m && j < n ) {
    v[i*n + j] = beta * v[i*n + j] + (1.0 - beta)*nabla[i*n + j];
    r[i*n + j] = gamma * r[i*n + j] + (1.0 - gamma)*nabla[i*n + j]*nabla[i*n + j];

    float v_hat = v[i*n + j] / (1.0 - beta_);
    float r_hat = r[i*n + j] / (1.0 - gamma_);
    update[i*n + j] = -EPS * v_hat / (sqrt(r_hat) + adam_eps);
  }
}

void cuda_adam_kernel ( const int m, const int n,
                        float* v, float* r,
                        float* update, float* nabla,
                        const float beta, const float gamma,
                        const float beta_, const float gamma_,
                        const float EPS, const float adam_eps ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num, thread_num, 1);
  
  adam<<<grid, block>>>( m, n, v, r, update, nabla, beta, gamma, beta_, gamma_, EPS, adam_eps  );
}