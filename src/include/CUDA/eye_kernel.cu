__global__
void eye ( int m, int n, float* A ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    if( i == j ) A[i*n + j] = 1.0f;
    else A[i*n + j] = 0.0f;
  }
}

void cuda_eye_kernel ( int m, int n, float* A ) {
  const int thread_num = 32;

  dim3 grid(16, 16, 1);
  dim3 block(thread_num, thread_num, 1);
  
  eye<<<grid, block>>>( m, n, A );
}