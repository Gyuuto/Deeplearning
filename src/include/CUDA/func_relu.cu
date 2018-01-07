__global__
void relu ( int m, int n, float* x ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    if( x[i*n + j] < 0.0f ) x[i*n + j] = 0.0f;
  }
}
__global__
void relu_diff ( int m, int n, float* x ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    if( x[i*n + j] < 0.0f ) x[i*n + j] = 0.0f;
    else x[i*n + j] = 1.0f;
  }
}

void cuda_func_relu_kernel ( int m, int n, float* x, bool isdiff ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num, thread_num, 1);

  if( isdiff ) relu_diff<<<grid, block>>>( m, n, x );
  else relu<<<grid, block>>>(m, n, x);
}
