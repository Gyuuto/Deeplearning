__global__
void clip ( float val, float* A, int m, int n ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    if( A[i * n + j] > val ) A[i * n + j] = val;
    else if( A[i * n + j] < -val ) A[i * n + j] = -val;
  }
}

void cuda_clip_kernel ( float val, float* A, int m, int n ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num, thread_num, 1);
  
  clip<<<grid, block>>>( val, A, m, n );
}