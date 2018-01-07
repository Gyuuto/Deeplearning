__global__
void grad_bias ( int m, int n, float* nabla_b, float* delta ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if( i < m ) {
    nabla_b[i] = 0.0f;
    for( int j = 0; j < n; ++j ) nabla_b[i] += delta[i*n + j];
  }
}

void cuda_full_grad_bias_kernel ( int m, int n, float* nabla_b, float* delta ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num*thread_num, 1, 1);
  
  grad_bias<<<grid, block>>>( m, n, nabla_b, delta  );
}