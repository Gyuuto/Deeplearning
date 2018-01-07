__global__
void grad_apply_bias ( const int m, const int n, const float* b, float* v ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  
  if( i < m && j < n ) {
    v[i*n + j] += b[i];
  }
}

void cuda_full_apply_bias_kernel ( int m, int n, float* b, float* v ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num, thread_num, 1);
  
  grad_apply_bias<<<grid, block>>>( m, n, b, v );
}