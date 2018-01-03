#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

// cudaMatrix
void cuda_eye_kernel( int m, int n, float* A );
void cuda_ones_kernel( int m, int n, float* A );
void cuda_zeros_kernel( int m, int n, float* A );
void cuda_hadamard_kernel( int m, int n, float* A, float* B, float* Y );
void cuda_sub_kernel( int y, int x, int h, int w, float* src, int ld_src, float* dest, int ld_dest );
void cuda_clip_kernel ( float val, float* A, int m, int n );
void cuda_hadamard_inplace_kernel( int m, int n, float* A, float* B );

// Function
void cuda_func_relu_kernel(int m, int n, float* x, bool isdiff);
void cuda_func_leakyrelu_kernel(int m, int n, float alpha, float* x, bool isdiff);
void cuda_func_sigmoid_kernel(int m, int n, float alpha, float* x, bool isdiff );
void cuda_func_tanh_kernel(int m, int n, float* x, bool isdiff);
void cuda_func_softsign_kernel( int m, int n, float* x, bool isdiff );
void cuda_func_softplus_kernel( int m, int n, float* x, bool isdiff );
void cuda_func_polynomial_kernel( int m, int n, int degree, float* x, bool isdiff );
void cuda_func_truncatedpower_kernel( int m, int n, int degree, float* x, bool isdiff );
void cuda_func_abs_kernel( int m, int n, float* x, bool isdiff );
void cuda_func_pow_kernel( int m, int n, int degree, float* x, bool isdiff );
void cuda_func_log_kernel( int m, int n, float* x, bool isdiff );
void cuda_func_exp_kernel( int m, int n, float* x, bool isdiff );
void cuda_func_softmax_kernel( int m, int n, float* x, bool isdiff );
void cuda_func_square_kernel( int m, int n, float* x, float* d, bool isdiff );
void cuda_func_crossentropy_kernel( int m, int n, float* x, float* d, bool isdiff);

// FullyConnected
void cuda_full_grad_bias_kernel( int nabla_b_m, float* nabla_b, float* delta, int ld_delta);
void cuda_full_apply_bias_kernel( int b_m, int n, float* b, float* ret);

#endif
