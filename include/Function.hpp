#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cmath>
#include <algorithm>

#include "Matrix.hpp"

#ifdef USE_GPU
#include "clMatrix.hpp"
#endif

template<typename T>
class Function
{
public:
	virtual Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const = 0;
#ifdef USE_GPU
	virtual clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const = 0;
#endif
};

template<typename T>
class LossFunction
{
public:
	virtual Matrix<T> operator() ( const Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const = 0;
#ifdef USE_GPU
	virtual clMatrix<T> operator() ( const clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const = 0;	
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

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			return clMatrix<T>::ones(x.m, x.n);
		}
		else{
			return x;
		}
	}
#endif
};

template<typename T>
class ReLU : public Function<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = x.v[i] <= 0.0 ? 0.0 : 1.0;
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::max(T(0.0), x.v[i]);
		}

		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_RELU_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_RELU_DIFF, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_RELU_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_RELU, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_RELU, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_RELU, y.m*y.n, 1 );
		}

		return y;
	}
#endif
};

template<typename T>
class LeakyReLU : public Function<T>
{
public:
	T alpha;
#ifdef USE_GPU
	cl_mem cl_alpha;
#endif
	
	LeakyReLU ( const T alpha = 0.2 ) :alpha(alpha)
	{
#ifdef USE_GPU
		cl_int err;
		cl_alpha = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(T), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_alpha, CL_TRUE, 0,
									sizeof(T), &(this->alpha), 0, NULL, NULL );
#endif
	}

#ifdef USE_GPU
	~LeakyReLU ()
	{
		clReleaseMemObject(cl_alpha);
	}
#endif
	
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = x.v[i] <= 0.0 ? alpha : 1.0;
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = (x.v[i] <= 0.0 ? alpha : 1.0)*x.v[i];
		}

		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU_DIFF, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU_DIFF, 2, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_LEAKYRELU_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_LEAKYRELU, 2, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_LEAKYRELU, y.m*y.n, 1 );
		}

		return y;
	}
#endif
};

template<typename T>
class Sigmoid : public Function<T>
{
public:
	T alpha;
#ifdef USE_GPU
	cl_mem cl_alpha;
#endif
	Sigmoid( T alpha = 1.0 ) :alpha(alpha)
	{
#ifdef USE_GPU
		cl_int err;
		cl_alpha = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(T), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_alpha, CL_TRUE, 0,
									sizeof(T), &alpha, 0, NULL, NULL );
#endif
	}

#ifdef USE_GPU
	~Sigmoid ()
	{
		clReleaseMemObject( cl_alpha );
	}
#endif
	
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
				T tmp = 1.0 + std::exp(-alpha*x.v[i]);
				y.v[i] = alpha*std::exp(-alpha*x.v[i]) / (tmp*tmp);
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = 1.0 / (1.0 + std::exp(-alpha*x.v[i]));
		}
		
		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);
		cl_int err;
		
		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID_DIFF, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID_DIFF, 2, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_SIGMOID_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SIGMOID, 2, &cl_alpha );
			cl_device_manager.run_kernel( PRG::FUNC_SIGMOID, y.m*y.n, 1 );
		}
		
		return y;
	}
#endif
};

template<typename T>
class Tanh : public Function<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
				T tmp = std::tanh(x.v[i]);
				y.v[i] = 1.0 - tmp*tmp;
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::tanh(x.v[i]);
		}
			
		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_TANH_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_TANH_DIFF, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_TANH_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_TANH, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_TANH, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_TANH, y.m*y.n, 1 );
		}
			
		return y;
	}
#endif
};

template<typename T>
class Softsign : public Function<T>
{
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
				T tmp = 1.0 + std::abs(x.v[i]);
				T y_diff = 0.0;
				if( x.v[i] > 1.0E-10 ) y_diff = 1.0;
				else if( x.v[i] < -1.0E-10 ) y_diff = -1.0;
				y.v[i] = (tmp - x.v[i]*y_diff)/(tmp*tmp);
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = x.v[i] / (1.0 + std::abs(x.v[i]));
		}
			
		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SOFTSIGN_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTSIGN_DIFF, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTSIGN_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_SOFTSIGN, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTSIGN, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTSIGN, y.m*y.n, 1 );
		}
			
		return y;
	}	
#endif
};
  
template<typename T>
class Softplus : public Function<T>
{
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
				T tmp = std::exp(x.v[i]);
				y.v[i] = tmp / (1.0 + tmp);
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::log(1.0 + std::exp(x.v[i]));
		}
			
		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_SOFTPLUS_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTPLUS_DIFF, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTPLUS_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_SOFTPLUS, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTPLUS, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTPLUS, y.m*y.n, 1 );
		}
			
		return y;
	}	
#endif

};

template<typename T, int n>
class Polynomial : public Function<T>
{
#ifdef USE_GPU

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
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = n*std::pow(x.v[i], n-1);
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::pow(x.v[i], n);
		}
			
		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL_DIFF, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL_DIFF, 2, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POLYNOMIAL_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POLYNOMIAL, 2, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POLYNOMIAL, y.m*y.n, 1 );
		}
			
		return y;
	}	
#endif
};

template<typename T, int n>
class TruncatedPower : public Function<T>
{
#ifdef USE_GPU
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
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = (x.v[i] < 0.0 ? 0.0 : n*std::pow(x.v[i], n-1));
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = (x.v[i] < 0.0 ? 0.0 : std::pow(x.v[i], n));
		}
			
		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER_DIFF, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER_DIFF, 2, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_TRUNCATEDPOWER_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_TRUNCATEDPOWER, 2, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_TRUNCATEDPOWER, y.m*y.n, 1 );
		}
			
		return y;
	}
#endif
};

template<typename T>
class Abs : public Function<T>
{
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
				T y_diff = 0.0;
				if( x.v[i] > 1.0E-10 ) y_diff = 1.0;
				else if( x.v[i] < -1.0E-10 ) y_diff = -1.0;
				y.v[i] = y_diff;
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::abs(x.v[i]);
		}
			
		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_ABS_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_ABS_DIFF, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_ABS_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_ABS, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_ABS, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_ABS, y.m*y.n, 1 );
		}
			
		return y;
	}
#endif
};

template<typename T>
class Pow : public Function<T>
{
public:
	T n;
#ifdef USE_GPU
	cl_mem cl_n;
#endif
	Pow ( T n = 1.0 ) :n(n)
	{
#ifdef USE_GPU
		cl_int err;
		cl_n = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(T), NULL, &err );
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_n, CL_TRUE, 0,
									sizeof(T), &n, 0, NULL, NULL );
#endif
	}
#ifdef USE_GPU
	~Pow ()
	{
		clReleaseMemObject(cl_n);
	}
#endif

	Matrix<T> operator () ( const Matrix<T>& x, const bool& isdiff ) const {
		Matrix<T> y(x.m, x.n);
		
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < y.m*y.n; ++i )
				y.v[i] = n*pow(x.v[i], n-1);
		}
		else{
#pragma omp parallel for
			for( int i = 0; i < y.m*y.n; ++i )
				y.v[i] = pow(x.v[i], n);
		}

		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_POW_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_POW_DIFF, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POW_DIFF, 2, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POW_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_POW, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_POW, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_POW, 2, &cl_n );
			cl_device_manager.run_kernel( PRG::FUNC_POW, y.m*y.n, 1 );
		}
			
		return y;
	}
#endif
};

template<typename T>
class Log : public Function<T>
{
public:
	Matrix<T> operator () ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);
		
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < y.m*y.n; ++i )
				y.v[i] = 1.0/x.v[i];
		}
		else{
#pragma omp parallel for
			for( int i = 0; i < y.m*y.n; ++i )
				y.v[i] = log(x.v[i]);
		}

		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_LOG_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_LOG_DIFF, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_LOG_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_LOG, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_LOG, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_LOG, y.m*y.n, 1 );
		}
			
		return y;
	}
#endif
};

template<typename T>
class Exp : public Function<T>
{
public:
	Matrix<T> operator () ( const Matrix<T>& x, const bool& isdiff ) const{
		Matrix<T> y(x.m, x.n);
		
		if( isdiff ){
#pragma omp parallel for
			for( int i = 0; i < y.m*y.n; ++i )
				y.v[i] = exp(x.v[i]);
		}
		else{
#pragma omp parallel for
			for( int i = 0; i < y.m*y.n; ++i )
				y.v[i] = exp(x.v[i]);
		}

		return y;
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		clMatrix<T> y(x.m, x.n);

		if( isdiff ){
			cl_device_manager.set_argument( PRG::FUNC_EXP_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_EXP_DIFF, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_EXP_DIFF, y.m*y.n, 1 );
		}
		else{
			cl_device_manager.set_argument( PRG::FUNC_EXP, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_EXP, 1, &x.v );
			cl_device_manager.run_kernel( PRG::FUNC_EXP, y.m*y.n, 1 );
		}
			
		return y;
	}
#endif
};


template<typename T>
class Softmax : public Function<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			return Matrix<T>::ones(x.m, x.n);
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
			
			Matrix<T> y(x.m, x.n);
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::exp(x.v[i] - max_val(0,i%y.n)) / sum(0,i%y.n);
			
			return y;
		}
	}

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const bool& isdiff ) const{
		if( isdiff ){
			return clMatrix<T>::ones(x.m, x.n);
		}
		else{
			clMatrix<T> sum(1, x.n), max_val(1, x.n);

			cl_int err;
			
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 0, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 1, &max_val.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 2, &sum.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 3, &x.M );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX_HELPER, 4, &x.N );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTMAX_HELPER, x.n, 1 );
			
			clMatrix<T> y(x.m, x.n);
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX, 2, &max_val.v );
			cl_device_manager.set_argument( PRG::FUNC_SOFTMAX, 3, &sum.v );
			cl_device_manager.run_kernel( PRG::FUNC_SOFTMAX, y.m, y.n );
			
			return y;
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
			Matrix<T> y(x.m, x.n);
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = 2.0*(x.v[i] - d.v[i]);

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

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			clMatrix<T> y(x.m, x.n);

			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 2, &d.v );
			cl_device_manager.run_kernel( PRG::FUNC_SQUARE_DIFF, y.m*y.n, 1 );

			return y;
		}
		else{
			clMatrix<T> y = x - d;

			y = clMatrix<T>::hadamard(y, y);
			
			for( int i = x.m*x.n; i > 0; i /= cl_device_manager.get_max_work_item(0) ){
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 0, &y.v );
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 1, cl_device_manager.get_max_work_item(0)*sizeof(T) );
				cl_device_manager.run_kernel( PRG::FUNC_SQUARE, i, 1 );
			}

			return y.sub(0, 0, 1, 1);
		}
	}
#endif
};

template<typename T>
class CrossEntropy : public LossFunction<T>
{
public:
	Matrix<T> operator() ( const Matrix<T>& x, const Matrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			Matrix<T> y(x.m, x.n);

#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) y.v[i] = 2.0*(x.v[i] - d.v[i]);

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

#ifdef USE_GPU
	clMatrix<T> operator() ( const clMatrix<T>& x, const clMatrix<T>& d, const bool& isdiff ) const{
		if( isdiff ){
			clMatrix<T> y(x.m, x.n);

			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_SQUARE_DIFF, 2, &d.v );
			cl_device_manager.run_kernel( PRG::FUNC_SQUARE_DIFF, y.m*y.n, 1 );

			return y;
		}
		else{
			clMatrix<T> y(x.m, x.n);

			cl_device_manager.set_argument( PRG::FUNC_CROSSENTROPY, 0, &y.v );
			cl_device_manager.set_argument( PRG::FUNC_CROSSENTROPY, 1, &x.v );
			cl_device_manager.set_argument( PRG::FUNC_CROSSENTROPY, 2, &d.v );
			cl_device_manager.run_kernel( PRG::FUNC_CROSSENTROPY, x.m*x.n, 1 );

			for( int i = x.m*x.n; i > 0; i /= cl_device_manager.get_max_work_item(0) ){
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 0, &y.v );
				cl_device_manager.set_argument( PRG::FUNC_SQUARE, 1, cl_device_manager.get_max_work_item(0)*sizeof(T) );
				cl_device_manager.run_kernel( PRG::FUNC_SQUARE, i, 1 );
			}
			
			clMatrix<T> ret = -2.0*y.sub(0, 0, 1, 1);

			return ret;
		}
	}
#endif
};
	
#endif
