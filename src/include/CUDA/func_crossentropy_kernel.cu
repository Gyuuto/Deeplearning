__global__
void crossentropy_helper ( int m, int n, float* x, const float* d ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) x[i*n + j] = -2.0f*d[i*n + j]*log(x[i*n + j]);
}
__global__
void crossentropy ( int n, float* x, const float* d ) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  float my_sum = (i < n ? x[i] : 0.0f);
  if( i + blockDim.x < n ) my_sum += x[i + blockDim.x];
  sdata[tid] = my_sum;
  __syncthreads();

  for( unsigned int s = blockDim.x/2; s > 32; s >>= 1 ) {
    if( tid < s ) {
      sdata[tid] = my_sum = my_sum + sdata[tid + s];
    }
    __syncthreads();
  }

  if( tid < 32 ) {
    if( blockDim.x >= 64 ) my_sum += sdata[tid + 32];
    for( int offset = 32/2; offset > 0; offset >>= 1 ) {
      my_sum += __shfl_down(my_sum, offset);
    }
  }
  if( tid == 0 ) x[blockIdx.x] = my_sum;
}
__global__
void crossentropy_diff ( int m, int n, float* x, const float* d ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < m && j < n ) {
    x[i*n + j] = 2.0f*(x[i*n + j] - d[i*n + j]);
  }
}

void cuda_func_crossentropy_kernel ( int m, int n, float* x, float* d, bool isdiff ) {
  const int thread_num = 32;

  if( isdiff ) {
    dim3 grid((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
    dim3 block(thread_num, thread_num, 1);
    crossentropy_diff<<<grid, block>>>( m, n, x, d );
  }
  else {
    dim3 grid_helper((m + thread_num)/thread_num, (n + thread_num)/thread_num, 1);
    dim3 block_helper(thread_num, thread_num, 1);
    crossentropy_helper<<<grid_helper, block_helper>>>(m, n, x, d);

    int blocks = (m*n + thread_num)/thread_num, tmp_n = m*n;
    int shared_mem_size = 2*thread_num*thread_num*sizeof(float);
    while( blocks > 1 ) {
      crossentropy<<<blocks, thread_num*thread_num, shared_mem_size>>>(tmp_n, x, d);
      tmp_n = blocks;
      blocks = (blocks - 1) / (2*thread_num*thread_num) + 1;
      cudaThreadSynchronize();
    }
    crossentropy<<<blocks, thread_num*thread_num, shared_mem_size>>>(tmp_n, x, d);
  }
}
