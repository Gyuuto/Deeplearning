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
	virtual inline Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d, const bool& isdiff ) = 0;
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

class Softsign : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j ){
					double tmp = 1.0 + std::abs(y(i,j));
					double y_diff = 0.0;
					if( y(i,j) > 1.0E-10 ) y_diff = 1.0;
					else if( y(i,j) < -1.0E-10 ) y_diff = -1.0;
					y(i,j) = (tmp - y(i,j)*y_diff)/(tmp*tmp);
				}
		}
		else{
			int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = y(i,j) / (1.0 + std::abs(y(i,j)));
		}
			
		return y;
	}	
};
  
class Softplus : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j ){
					double tmp = std::exp(y(i,j));
					y(i,j) = tmp / (1.0 + tmp);
				}
		}
		else{
			int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = std::log(1.0 + std::exp(y(i,j)));
		}
			
		return y;
	}	
};

template<int n>
class Polynomial : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j ){
					y(i,j) = n*std::pow(y(i,j), n-1);
				}
		}
		else{
			int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = std::pow(y(i,j), n);
		}
			
		return y;
	}	
};

template<int n>
class TruncatedPower : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j ){
					y(i,j) = (y(i,j) < 0.0 ? 0.0 : n*std::pow(y(i,j), n-1));
				}
		}
		else{
			int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = (y(i,j) < 0.0 ? 0.0 : std::pow(y(i,j), n));
		}
			
		return y;
	}	
};

class Abs : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		auto y = x;

		if( isdiff ){
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j ){
					double y_diff = 0.0;
					if( y(i,j) > 1.0E-10 ) y_diff = 1.0;
					else if( y(i,j) < -1.0E-10 ) y_diff = -1.0;
					y(i,j) = y_diff;
				}
		}
		else{
			int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(y)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = std::abs(y(i,j));
		}
			
		return y;
	}	
};

class Softmax : public Function
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		if( isdiff ){
			return Matrix<double>::ones(x.m, x.n);
		}
		else{
			Matrix<double> sum(1, x.n), max_val(1, x.n);
			
			for( int i = 0; i < x.n; ++i ){
				sum(0,i) = 0.0;
				max_val(0,i) = x(0,i);
				for( int j = 0; j < x.m; ++j )
					max_val(0,i) = std::max(max_val(0,i), x(j,i));
				for( int j = 0; j < x.m; ++j )
					sum(0,i) += std::exp(x(j,i) - max_val(0,i));
			}
			
			Matrix<double> y(x.m, x.n);
			int i, j;
#pragma omp parallel for default(none)			\
	private(i, j), shared(x, y, sum, max_val)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) = std::exp(x(i,j) - max_val(0,j)) / sum(0,j);
			
			return y;
		}
	}
};

///////////////////////////////////////////////////////
//////////////////// Loss function ////////////////////
///////////////////////////////////////////////////////
class Square : public LossFunction
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d, const bool& isdiff ){
		if( isdiff ){
			auto y = x;
			int i, j;
#pragma omp parallel for default(none)			\
	private(i,j), shared(y, d)
			for( i = 0; i < y.m; ++i )
				for( j = 0; j < y.n; ++j )
					y(i,j) -= d(i,j);
			return y;
		}
		else{
			Matrix<double> y(1, 1);
			double y_ = 0.0;

			int i, j;
#pragma omp parallel for default(none) reduction(+:y_)	\
	private(i,j), shared(x, d, y)
			for( i = 0; i < x.m; ++i )
				for( j = 0; j < x.n; ++j ){
					double tmp = x(i,j) - d(i,j);
					y_ += tmp*tmp;
				}
			y(0,0) = y_;
			return y;
		}
	}
};

class CrossEntropy : public LossFunction
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d, const bool& isdiff ){
		if( isdiff ){
			Matrix<double> y(x.m, x.n);
			int i, j;

#pragma omp parallel for default(none)	\
	private(i, j), shared(y, x, d)
			for( i = 0; i < x.m; ++i )
				for( j = 0; j < x.n; ++j )
					y(i,j) = x(i,j) - d(i,j);

			return y;
		}
		else{
			double y_ = 0.0;
			Matrix<double> y(1,1);

			int i, j;
#pragma omp parallel for default(none) reduction(+:y_)	\
	private(i,j), shared(d, x)
			for( i = 0; i < x.m; ++i )
				for( j = 0; j < x.n; ++j )
					y_ -= d(i,j)*std::log(x(i,j));

			y(0,0) = y_;
			return 2.0*y;
		}
	}
};
	
#endif
