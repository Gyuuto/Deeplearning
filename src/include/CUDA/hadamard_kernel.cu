__global__
void hadamard ( int m, int n, float* A, float* B, float* Y ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    Y[i*n + j] = A[i*n + j]*B[i*n + j];
  }
}

void cuda_hadamard_kernel ( int m, int n, float* A, float* B, float* Y ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num, thread_num, 1);
  
  hadamard<<<grid, block>>>( m, n, A, B, Y );
}