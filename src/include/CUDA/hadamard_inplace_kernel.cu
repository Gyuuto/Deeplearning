__global__
void hadamard_inplace ( int m, int n, float* A, float* B ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    A[i*n + j] = A[i*n + j]*B[i*n + j];
  }
}

void cuda_hadamard_inplace_kernel ( int m, int n, float* A, float* B ) {
  const int thread_num = 32;

  dim3 grid(16, 16, 1);
  dim3 block(thread_num, thread_num, 1);
  
  hadamard_inplace<<<grid, block>>>( m, n, A, B );
}