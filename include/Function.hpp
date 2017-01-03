#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <cmath>
#include <algorithm>

#include "Matrix.hpp"

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
		Matrix<double> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = x.v[i] <= 0.0 ? 0.0 : 1.0;
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::max(0.0, x.v[i]);
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
		Matrix<double> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
				double tmp = 1.0 + std::exp(-alpha*x.v[i]);
				y.v[i] = alpha*std::exp(-alpha*x.v[i]) / (tmp*tmp);
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = 1.0 / (1.0 + std::exp(-alpha*x.v[i]));
		}
		
		return y;
	}
};

class Tanh : public Function
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		Matrix<double> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
				double tmp = std::tanh(x.v[i]);
				y.v[i] = 1.0 - tmp*tmp;
			}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::tanh(x.v[i]);
		}
			
		return y;
	}
};

class Softsign : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		Matrix<double> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
					double tmp = 1.0 + std::abs(x.v[i]);
					double y_diff = 0.0;
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
};
  
class Softplus : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		Matrix<double> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
					double tmp = std::exp(x.v[i]);
					y.v[i] = tmp / (1.0 + tmp);
				}
		}
		else{
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::log(1.0 + std::exp(x.v[i]));
		}
			
		return y;
	}	
};

template<int n>
class Polynomial : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		Matrix<double> y(x.m, x.n);

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
};

template<int n>
class TruncatedPower : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		Matrix<double> y(x.m, x.n);

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
};

class Abs : public Function
{
	inline Matrix<double> operator() ( const Matrix<double>& x, const bool& isdiff ){
		Matrix<double> y(x.m, x.n);

		if( isdiff ){
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ){
					double y_diff = 0.0;
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

#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.n; ++i ){
				sum(0,i) = 0.0;
				max_val(0,i) = x(0,i);
				for( int j = 0; j < x.m; ++j )
					max_val(0,i) = std::max(max_val(0,i), x(j,i));
				for( int j = 0; j < x.m; ++j )
					sum(0,i) += std::exp(x(j,i) - max_val(0,i));
			}
			
			Matrix<double> y(x.m, x.n);
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = std::exp(x.v[i] - max_val(0,i%y.n)) / sum(0,i%y.n);
			
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
			Matrix<double> y(x.m, x.n);
#pragma omp parallel for schedule(auto)
			for( int i = 0; i < y.m*y.n; ++i ) y.v[i] = x.v[i] - d.v[i];

			return y;
		}
		else{
			Matrix<double> y(1, 1);
			double y_ = 0.0;

#pragma omp parallel for schedule(auto) reduction(+:y_)
			for( int i = 0; i < x.m*x.n; ++i ){
				double tmp = x.v[i] - d.v[i];
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

#pragma omp parallel for schedule(auto)
			for( int i = 0; i < x.m*x.n; ++i ) y.v[i] = x.v[i] - d.v[i];

			return y;
		}
		else{
			double y_ = 0.0;
			Matrix<double> y(1,1);

#pragma omp parallel for schedule(auto) reduction(-:y_)
			for( int i = 0; i < x.m*x.n; ++i ) y_ -= d.v[i]*std::log(x.v[i]);

			y(0,0) = y_;
			return 2.0*y;
		}
	}
};
	
#endif
