__global__
void zeros ( int m, int n, float* A ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    A[i*n + j] = 0.0f;
  }
}

void cuda_zeros_kernel ( int m, int n, float* A ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num, thread_num, 1);
  
  zeros<<<grid, block>>>( m, n, A );
}