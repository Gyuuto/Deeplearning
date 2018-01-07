#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cmath>
#include <algorithm>

#include <Matrix.hpp>

#ifdef USE_OPENCL
#include <clMatrix.hpp>
#endif
#ifdef USE_CUDA
#include <cudaMatrix.hpp>
#endif

template<typename T>
class Function
{
public:
	virtual Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const = 0;
	virtual void inplace ( Matrix<T>& x, const bool& isdiff ) const = 0;
#ifdef USE_OPENCL
	virtual clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const = 0;
	virtual void inplace ( clMatrix<T>& x, const bool& isdiff ) const = 0;
#endif
#ifdef USE_CUDA
	virtual cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const = 0;
	virtual void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const = 0;
#endif
};

template<typename T>
class LossFunction
{
public:
	virtual Matrix<T> operator() ( const Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const = 0;
	virtual void inplace ( Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const = 0;
#ifdef USE_OPENCL
	virtual clMatrix<T> operator() ( const clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const = 0;
	virtual void inplace ( clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const = 0;	
#endif
#ifdef USE_CUDA
	virtual cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const cudaMatrix<T>& d, const bool& isdiff ) const = 0;
	virtual void inplace ( cudaMatrix<T>& x, const cudaMatrix<T>& d, const bool& isdiff ) const = 0;
#endif
};

template<typename T>
class Identity : public Function<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			return Matrix<T>::ones(x.m, x.n);
		}
		else{
			return x;
		}
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = 1.0;
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			return clMatrix<T>::ones(x.m, x.n);
		}
		else{
			return x;
		}
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::CLMAT_ONES, 0, &x.v );
			cl_device_manager.run_kernel( PRG::CLMAT_ONES, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			return cudaMatrix<T>::ones(x.m, x.n);
		}
		else{
			return x;
		}
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
            cuda_ones_kernel( x.m, x.n, x.v );
		}
	}
#endif
};

template<typename T>
class ReLU : public Function<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = (x.v[i] <= 0.0 ? 0.0 : 1.0);
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = std::max(T(0.0), x.v[i]);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_RELU_DIFF, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_RELU_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_RELU, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_RELU, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_relu_kernel(x.m, x.n, x.v, isdiff);
	}
#endif
};

template<typename T>
class LeakyReLU : public Function<T>
{
public:
	T alpha;
#ifdef USE_OPENCL
	cl_mem cl_alpha;
#endif
	
	LeakyReLU ( const T alpha = 0.2 ) :alpha(alpha)
	{
#ifdef USE_OPENCL
		cl_int err;
		cl_alpha = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(T), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_alpha, CL_TRUE, 0,
									sizeof(T), &(this->alpha), 0, NULL, NULL );
#endif
	}

#ifdef USE_OPENCL
	~LeakyReLU ()
	{
		clReleaseMemObject(cl_alpha);
	}
#endif
	
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = x.v[i] <= 0.0 ? alpha : 1.0;
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = (x.v[i] <= 0.0 ? alpha : 1.0)*x.v[i];
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU_DIFF, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU_DIFF, 1, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_LEAKYRELU_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU, 1, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_LEAKYRELU, x.m*x.n, 1 );
		}
	}	 
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_leakyrelu_kernel(x.m, x.n, alpha, x.v, isdiff);
	}	 
#endif
};

template<typename T>
class Sigmoid : public Function<T>
{
public:
	T alpha;
#ifdef USE_OPENCL
	cl_mem cl_alpha;
#endif
	Sigmoid( T alpha = 1.0 ) :alpha(alpha)
	{
#ifdef USE_OPENCL
		cl_int err;
		cl_alpha = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(T), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_alpha, CL_TRUE, 0,
									sizeof(T), &alpha, 0, NULL, NULL );
#endif
	}

#ifdef USE_OPENCL
	~Sigmoid ()
	{
		clReleaseMemObject( cl_alpha );
	}
#endif
	
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);
		
		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ){
				T tmp = 1.0 + std::exp(-alpha*x.v[i]);
				x.v[i] = alpha*std::exp(-alpha*x.v[i]) / (tmp*tmp);
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = 1.0 / (1.0 + std::exp(-alpha*x.v[i]));
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID_DIFF, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID_DIFF, 1, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_SIGMOID_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID, 1, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_SIGMOID, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_sigmoid_kernel( x.m, x.n, alpha, x.v, isdiff );
	}
#endif
};

template<typename T>
class Tanh : public Function<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ){
				T tmp = std::tanh(x.v[i]);
				x.v[i] = 1.0 - tmp*tmp;
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = std::tanh(x.v[i]);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_TANH_DIFF, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_TANH_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_TANH, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_TANH, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_tanh_kernel(x.m, x.n, x.v, isdiff);
	}
#endif
};

template<typename T>
class Softsign : public Function<T>
{
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ){
				T tmp = 1.0 + std::abs(x.v[i]);
				T x_diff = 0.0;
				if( x.v[i] > 1.0E-10 ) x_diff = 1.0;
				else if( x.v[i] < -1.0E-10 ) x_diff = -1.0;
				x.v[i] = (tmp - x.v[i]*x_diff)/(tmp*tmp);
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = x.v[i] / (1.0 + std::abs(x.v[i]));
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}	
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SOFTSIGN_DIFF, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTSIGN_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_SOFTSIGN, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTSIGN, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}	
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_softsign_kernel( x.m, x.n, x.v, isdiff );
	}
#endif
};
  
template<typename T>
class Softplus : public Function<T>
{
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ){
				T tmp = std::exp(x.v[i]);
				x.v[i] = tmp / (1.0 + tmp);
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = std::log(1.0 + std::exp(x.v[i]));
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}	
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SOFTPLUS_DIFF, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTPLUS_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_SOFTPLUS, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTPLUS, x.m*x.n, 1 );
		}
	}	
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}	
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_softplus_kernel( x.m, x.n, x.v, isdiff );
	}	
#endif
};

template<typename T, int n>
class Polynomial : public Function<T>
{
#ifdef USE_OPENCL

	cl_mem cl_n;
	Polynomial()
	{
		cl_int err;
		int val_n = n;
		
		cl_n = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_n, CL_TRUE, 0,
									sizeof(int), &val_n, 0, NULL, NULL );
	}

	~Polynomial ()
	{
		clReleaseMemObject(cl_n);
	}
#endif
	
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = n*std::pow(x.v[i], n-1);
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = std::pow(x.v[i], n);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}	
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL_DIFF, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL_DIFF, 1, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POLYNOMIAL_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL, 1, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POLYNOMIAL, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}	
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_polynomial_kernel( x.m, x.n, n, x.v, isdiff );
	}
#endif
};

template<typename T, int n>
class TruncatedPower : public Function<T>
{
#ifdef USE_OPENCL
	cl_mem cl_n;
	TruncatedPower()
	{
		cl_int err;
		int val_n = n;
		
		cl_n = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_n, CL_TRUE, 0,
									sizeof(int), &val_n, 0, NULL, NULL );
	}

	~TruncatedPower()
	{
		clReleaseMemObject(cl_n);
	}	
#endif

	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = (x.v[i] < 0.0 ? 0.0 : n*std::pow(x.v[i], n-1));
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = (x.v[i] < 0.0 ? 0.0 : std::pow(x.v[i], n));
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER_DIFF, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER_DIFF, 1, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_TRUNCATEDPOWER_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER, 1, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_TRUNCATEDPOWER, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_truncatedpower_kernel( x.m, x.n, n, x.v, isdiff );
	}
#endif
};

template<typename T>
class Abs : public Function<T>
{
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ){
				T x_diff = 0.0;
				if( x.v[i] > 1.0E-10 ) x_diff = 1.0;
				else if( x.v[i] < -1.0E-10 ) x_diff = -1.0;
				x.v[i] = x_diff;
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = std::abs(x.v[i]);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_ABS_DIFF, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_ABS_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_ABS, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_ABS, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_abs_kernel( x.m, x.n, x.v, isdiff );
	}
#endif
};

template<typename T>
class Pow : public Function<T>
{
public:
	T n;
#ifdef USE_OPENCL
	cl_mem cl_n;
#endif
	Pow ( T n = 1.0 ) :n(n)
	{
#ifdef USE_OPENCL
		cl_int err;
		cl_n = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(T), NULL, &err );
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_n, CL_TRUE, 0,
									sizeof(T), &n, 0, NULL, NULL );
#endif
	}
#ifdef USE_OPENCL
	~Pow ()
	{
		clReleaseMemObject(cl_n);
	}
#endif

	Matrix<T> operator () ( const Matrix<T>& x, const bool& isdiff ) const {
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const {
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i )
				x.v[i] = n*pow(x.v[i], n-1);
		}
		else{
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i )
				x.v[i] = pow(x.v[i], n);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_POW_DIFF, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POW_DIFF, 1, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POW_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_POW, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POW, 1, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POW, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_pow_kernel( x.m, x.n, n, x.v, isdiff );
	}
#endif
};

template<typename T>
class Log : public Function<T>
{
public:
	Matrix<T> operator () ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i )
				x.v[i] = 1.0/x.v[i];
		}
		else{
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i )
				x.v[i] = log(x.v[i]);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_LOG_DIFF, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_LOG_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_LOG, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_LOG, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_log_kernel( x.m, x.n, x.v, isdiff );
	}
#endif
};

template<typename T>
class Exp : public Function<T>
{
public:
	Matrix<T> operator () ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i )
				x.v[i] = exp(x.v[i]);
		}
		else{
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i )
				x.v[i] = exp(x.v[i]);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_EXP_DIFF, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_EXP_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_EXP, 0, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_EXP, x.m*x.n, 1 );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        cuda_func_exp_kernel( x.m, x.n, x.v, isdiff );
	}
#endif
};


template<typename T>
class Softmax : public Function<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = 1.0;
		}
		else{
			Matrix<T> sum(1, x.n), max_val(1, x.n);

#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.n; ++i ){
				sum(0,i) = 0.0;
				max_val(0,i) = x(0,i);
				for( int j = 0; j < x.m; ++j )
					max_val(0,i) = std::max(max_val(0,i), x(j,i));
				for( int j = 0; j < x.m; ++j )
					sum(0,i) += std::exp(x(j,i) - max_val(0,i));
			}
			
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = std::exp(x.v[i] - max_val(0,i%x.n)) / sum(0,i%x.n);
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::CLMAT_ONES, 0, &x.v );
			cl_device_manager.run_kernel( PRG::CLMAT_ONES, x.m*x.n, 1 );
		}
		else{
			clMatrix<T> sum(1, x.n), max_val(1, x.n);

			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 1, &max_val.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 2, &sum.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 3, &x.M );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 4, &x.N );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTMAX_HELPER, x.n, 1 );
			
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX, 1, &max_val.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX, 2, &sum.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTMAX, x.m, x.n );
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const bool& isdiff ) const{
		cudaMatrix<T> y = x;

		inplace(y, isdiff);

		return y;
	}
	void inplace ( cudaMatrix<T>& x, const bool& isdiff ) const{
        if( isdiff ) {
            cuda_func_softmax_kernel(x.m, x.n, x.v, NULL, NULL, isdiff);
        }
        else {
            cudaMatrix<T> sum(1, x.n), max_val(1, x.n);
            cuda_func_softmax_kernel(x.m, x.n, x.v, max_val.v, sum.v, isdiff);
        }
	}
#endif
};

///////////////////////////////////////////////////////
//////////////////// Loss function ////////////////////
///////////////////////////////////////////////////////
template<typename T>
class Square : public LossFunction<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			Matrix<T> y = x;

			inplace(y, d, isdiff);

			return y;
		}
		else{
			Matrix<T> y(1, 1);
			T y_ = 0.0;

#pragma omp parallel for schedule(auto) reduction(+:y_)
			for( int i = 0; i < x.m*x.n; ++i ){
				T tmp = x.v[i] - d.v[i];
				y_ += tmp*tmp;
			}

			y(0,0) = y_;
			return y;
		}
	}
	void inplace ( Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = 2.0*(x.v[i] - d.v[i]);
		}
		else{
			T y_ = 0.0;

#pragma omp parallel for schedule(auto) reduction(+:y_)
			for( int i = 0; i < x.m*x.n; ++i ){
				T tmp = x.v[i] - d.v[i];
				y_ += tmp*tmp;
			}

			x(0,0) = y_;
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			clMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y;
		}
		else{
			clMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y.sub(0, 0, 1, 1);
		}
	}
	void inplace ( clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 1, &d.v );
			cl_device_manager.run_kernel( PRG::FUNC_SQUARE_DIFF, x.m*x.n, 1 );
		}
		else{
			x -= d;
			x.hadamard(x);
			
			for( int i = x.m*x.n; i > 0; i /= cl_device_manager.get_max_work_item(0) ){
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 0, &x.v );
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 1, cl_device_manager.get_max_work_item(0)*sizeof(T) );
				cl_device_manager.run_kernel( PRG::FUNC_SQUARE, i, 1 );
			}
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const cudaMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			cudaMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y;
		}
		else{
			cudaMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y.sub(0, 0, 1, 1);
		}
	}
	void inplace ( cudaMatrix<T>& x, const cudaMatrix<T>& d, const bool& isdiff ) const{
        cuda_func_square_kernel( x.m, x.n, x.v, d.v, isdiff );
		// if( isdiff ){
		// 	cuda_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 0, &x.v );
		// 	cuda_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 1, &d.v );
		// 	cuda_device_manager.run_kernel( PRG::FUNC_SQUARE_DIFF, x.m*x.n, 1 );
		// }
		// else{
		// 	x -= d;
		// 	x.hadamard(x);
			
		// 	for( int i = x.m*x.n; i > 0; i /= cuda_device_manager.get_max_work_item(0) ){
		// 		cuda_device_manager.set_argument( PRG::FUNC_SQUARE, 0, &x.v );
		// 		cuda_device_manager.set_argument( PRG::FUNC_SQUARE, 1, cuda_device_manager.get_max_work_item(0)*sizeof(T) );
		// 		cuda_device_manager.run_kernel( PRG::FUNC_SQUARE, i, 1 );
		// 	}
		// }
	}
#endif
};

template<typename T>
class CrossEntropy : public LossFunction<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			Matrix<T> y = x;

			inplace(y, d, isdiff);

			return y;
		}
		else{
			T y_ = 0.0;
			Matrix<T> y(1,1);

#pragma omp parallel for schedule(auto) reduction(-:y_)
			for( int i = 0; i < x.m*x.n; ++i ) y_ -= d.v[i]*std::log(x.v[i]);

			y(0,0) = y_;
			return 2.0*y;
		}
	}
	void inplace ( Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) x.v[i] = 2.0*(x.v[i] - d.v[i]);
		}
		else{
			T y_ = 0.0;

#pragma omp parallel for schedule(auto) reduction(-:y_)
			for( int i = 0; i < x.m*x.n; ++i ) y_ -= d.v[i]*std::log(x.v[i]);

			x(0,0) = 2.0*y_;
		}
	}
#ifdef USE_OPENCL
	clMatrix<T> operator() ( const clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			clMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y;
		}
		else{
			clMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y.sub(0, 0, 1, 1);
		}
	}
	void inplace ( clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 1, &d.v );
			cl_device_manager.run_kernel( PRG::FUNC_SQUARE_DIFF, x.m*x.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_CROSSENTROPY, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_CROSSENTROPY, 1, &d.v );
			cl_device_manager.run_kernel( PRG::FUNC_CROSSENTROPY, x.m*x.n, 1 );

			for( int i = x.m*x.n; i > 0; i /= cl_device_manager.get_max_work_item(0) ){
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 0, &x.v );
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 1, cl_device_manager.get_max_work_item(0)*sizeof(T) );
				cl_device_manager.run_kernel( PRG::FUNC_SQUARE, i, 1 );
			}
			
			x *= -2.0;
		}
	}
#endif
#ifdef USE_CUDA
	cudaMatrix<T> operator() ( const cudaMatrix<T>& x, const cudaMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			cudaMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y;
		}
		else{
			cudaMatrix<T> y = x;

			inplace(y, d, isdiff);

			return y.sub(0, 0, 1, 1);
		}
	}
	void inplace ( cudaMatrix<T>& x, const cudaMatrix<T>& d, const bool& isdiff ) const{
        cuda_func_crossentropy_kernel(x.m, x.n, x.v, d.v, isdiff);
    }
#endif
};
	
#endif
