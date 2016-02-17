#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cmath>
#include <algorithm>

#include "matrix.hpp"

class Function
{
public:
	virtual inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ) = 0;
};

class LossFunction
{
public:
	// for normal apply
	virtual inline Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d, const bool& isdiff ) = 0;
};

class ReLU : public Function
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = (y(i,j) <= 0.0 ? 0.0 : 1.0);
		
		}
		else{
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = std::max(0.0, y(i,j));
		}

		return y;
	}
};
 
class Identity : public Function
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		if( isdiff ){
			return Matrix<double>::ones(x.m, x.n);
		}
		else{
			return x;
		}
	}
};

class Sigmoid : public Function
{
public:
	double alpha;
	Sigmoid( double alpha = 1.0 ) :alpha(alpha) {}
	
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j ){
					double tmp = 1.0 + std::exp(-alpha*y(i,j));
					y(i,j) = alpha*std::exp(-alpha*y(i,j)) / (tmp*tmp);
				}
		}
		else{
			int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = 1.0 / (1.0 + std::exp(-alpha*y(i,j)));
		}
		
		return y;
	}
};

class Tanh : public Function
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j ){
					double tmp = std::tanh(y(i,j));
					y(i,j) = 1.0 - tmp*tmp;
				}
		}
		else{
			int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = std::tanh(y(i,j));
		}
			
		return y;
	}
};

class Softmax : public Function
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		if( isdiff ){
			auto y = (*this)(x, false);
		
			int i, j;
#pragma omp parallel for default(none)			\
	private(i, j), shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = y(i,j)*(1.0 - y(i,j));
			return y;
		}
		else{
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
#pragma omp parallel for default(none)			\
	private(i, j), shared(y, sum)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = std::exp(y(i,j)) / sum(1,j);
			return y;
		}
	}
};

// Loss function
class SQER : public LossFunction
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d, const bool& isdiff ){
		if( isdiff ){
			auto y = x;
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j), shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) -= d(i,j);
			return y;
		}
		else{
			Matrix<double> y(1, 1);
			double y_ = 0.0;

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
	}
};

class CEER : public LossFunction
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d, const bool& isdiff ){
		if( isdiff ){
			auto y = x;

			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j), shared(y)
			for( i = 0; i < y.n; ++i )
				y(i,j) = d(i,j)*1.0/y(i,j);
		
			return -1.0*y;
		}
		else{
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
	}
};
	
#endif
