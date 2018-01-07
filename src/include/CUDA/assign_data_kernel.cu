__global__
void assign_data ( const int N, const int m, const int n, float* y, const float* x, const int* idx, const int offset ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  
  if( i < m && j < n ) {
    int tmp_idx = (j + offset)%N;
    y[i*n + j] = x[i*N + idx[tmp_idx]];
  }
}

void cuda_assign_data_kernel ( const int N, const int m, const int n, float* y, const float* x, const int* idx, const int offset ) {
  const int thread_num = 32;

  dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
  dim3 block(thread_num, thread_num, 1);
  
  assign_data<<<grid, block>>>( N, m, n, y, x, idx, offset );
}