__global__
void sub ( int y, int x, int h, int w, float* src, int ld_src, float* dest, int ld_dest ) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if( i < h && j < w ) {
    dest[i * ld_dest + j] = src[(i + y) * ld_src + (j + x)];
  }
}

void cuda_sub_kernel ( int y, int x, int h, int w, float* src, int ld_src, float* dest, int ld_dest ) {
  const int thread_num = 32;

  dim3 grid(16, 16, 1);
  dim3 block(thread_num, thread_num, 1);
  
  sub<<<grid, block>>>( y, x, h, w, src, ld_src, dest, ld_dest );
}