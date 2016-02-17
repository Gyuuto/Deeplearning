#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cmath>
#include <algorithm>

#include "matrix.hpp"

class Function
{
public:
	// for normal apply
	virtual Matrix<double> operator() ( const Matrix<double>& x ) = 0;
	// for differential apply
	virtual Matrix<double> operator[] ( const Matrix<double>& x ) = 0;
};

class LossFunction
{
public:
	// for normal apply
	virtual Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d ) = 0;
	// for differential apply
	virtual Matrix<double> operator[] ( const Matrix<double>& x, const Matrix<double>& d ) = 0;
};

class ReLU : public Function
{
public:
	Matrix<double> operator() ( const Matrix<double>& x ){
		auto y = x;

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y(i,j) = std::max(0.0, y(i,j));
		
		return y;
	}

	Matrix<double> operator[] ( const Matrix<double>& x ){
		auto y = x;

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y(i,j) = (y(i,j) <= 0.0 ? 0.0 : 1.0);
		
		return y;
	}
};
 
class Identity : public Function
{
public:
	Matrix<double> operator() ( const Matrix<double>& x ){
		return x;
	}
	Matrix<double> operator[] ( const Matrix<double>& x ){
		return Matrix<double>::ones(x.m, x.n);
	}
};

class Sigmoid : public Function
{
public:
	double alpha;
	Sigmoid( double alpha = 1.0 ) :alpha(alpha) {}
	
	Matrix<double> operator() ( const Matrix<double>& x ){
		auto y = x;

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y(i,j) = 1.0 / (1.0 + std::exp(-alpha*y(i,j)));
		
		return y;
	}
	Matrix<double> operator[] ( const Matrix<double>& x ){
		auto y = x;

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j ){
				double tmp = 1.0 + std::exp(-alpha*y(i,j));
				y(i,j) = alpha*std::exp(-alpha*y(i,j)) / (tmp*tmp);
			}
		
		return y;
	}
};

class Tanh : public Function
{
public:
	Matrix<double> operator() ( const Matrix<double>& x ){
		auto y = x;

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y(i,j) = std::tanh(y(i,j));
		
		return y;
	}
	Matrix<double> operator[] ( const Matrix<double>& x ){
		auto y = x;

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j ){
				double tmp = std::tanh(y(i,j));
				y(i,j) = 1.0 - tmp*tmp;
			}
		
		return y;
	}
};

class Softmax : public Function
{
public:
	Matrix<double> operator() ( const Matrix<double>& x ){
		Matrix<double> sum(1, x.n);
		
		for( int i = 0; i < x.n; ++i ){
			int j;
			double sum_ = 0.0;
#pragma omp parallel for default(none) reduction(+, sum_)	\
	private(i, j), shared(x, sum)
			for( j = 0; j < x.m; ++j )
				sum_ += std::exp(x(j,i));

			sum(1,i) = sum_;
		}

		auto y = x;
		int i, j;
#pragma omp parallel for default(none) \
		private(i, j), shared(y, sum)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y(i,j) = std::exp(y(i,j)) / sum(1,j);

		return y;
	}
	Matrix<double> operator[] ( const Matrix<double>& x ){
		auto y = *this(x);
		
#pragma omp parallel for default(none) \
		private(i, j), shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y(i,j) = y(i,j)*(1.0 - y(i,j));

		return y;
	}
};

// Loss function
class SQER : public LossFunction
{
public:
	Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d ){
		double y_ = 0.0;
		Matrix<double> y(1,1);

		int i, j;
#pragma omp parallel for default(none) reduction(+,y_)	\
	private(i,j), shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j ){
				double tmp = x(i,j) - d(i,j);
				y_ += tmp*tmp;
			}
		y(0,0) = y_;
		return 0.5*y;
	}
	Matrix<double> operator[] ( const Matrix<double>& x, const Matrix<double>& d ){
		Matrix<double> y(x.m, x.n);

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j), shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y(i,j) = y(i,j) - d(i,j);
		
		return y;
	}
};

class CEER : public LossFunction
{
public:
	Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d ){
		double y_ = 0.0;
		Matrix<double> y(1,1);

		int i, j;
#pragma omp parallel for default(none) reduction(+,y_)	\
	private(i,j), shared(y)
		for( i = 0; i < y.m; ++i )
			for( j = 0; j < y.n; ++j )
				y_ += d(i,j)*std::log(y(i,j));

		y(0,0) = -y_;
		return y;
	}
	Matrix<double> operator[] ( const Matrix<double>& x, const Matrix<double>& d ){
		auto y = x;

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j), shared(y)
		for( i = 0; i < y.n; ++i )
			y(i,j) = d(i,j)*1.0/y(i,j);
		
		return -1.0*y;
	}
};
	
#endif
