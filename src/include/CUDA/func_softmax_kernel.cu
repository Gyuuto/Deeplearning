__global__
void softmax_helper ( int m, int n, float* max_x, float* sum_x, const float* x ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if( i < n ) {
    max_x[i] = x[i];
    for( int j = 0; j < m; ++j )
      if( max_x[i] < x[j*n + i] ) max_x[i] = x[j*n + i];

    sum_x[i] = 0.0f;
    for( int j = 0; j < m; ++j ) sum_x[i] += exp(x[j*n + i] - max_x[i]);
  }
}
__global__
void softmax ( int m, int n, float* x, const float* max_x, const float* sum_x ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    x[i*n + j] = exp(x[i*n + j] - max_x[j]) / sum_x[j];
  }
}
__global__
void softmax_diff ( int m, int n, float* x ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    x[i*n + j] = 1.0f;
  }
}

void cuda_func_softmax_kernel ( int m, int n, float* x, float* max_x, float* sum_x, bool isdiff ) {
  const int thread_num = 32;

  if( isdiff ) {
    dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
    dim3 block(thread_num, thread_num, 1);
    softmax_diff<<<grid, block>>>( m, n, x );
  }
  else {
    dim3 helper_grid((m*n + thread_num)/thread_num, 1, 1);
    dim3 helper_block(thread_num*thread_num, 1, 1);
    softmax_helper<<<helper_grid, helper_block>>>( m, n, max_x, sum_x, x );

    dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
    dim3 block(thread_num, thread_num, 1);
    softmax<<<grid, block>>>(m, n, x, max_x, sum_x);
  }
}
