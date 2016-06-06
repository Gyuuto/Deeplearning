#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>

#include <assert.h>

#ifdef USE_EIGEN
#include <Eigen>
#elif USE_BLAS
extern "C"{
	void dgemm_(char* transa, char* transb, int* m, int* n, int* k,
				double* alpha, double* A, int* lda, double* B, int* ldb,
				double* beta, double* C, int* ldc);

	void sgemm_(char* transa, char* transb, int* m, int* n, int* k,
				float* alpha, float* A, int* lda, float* B, int* ldb,
				float* beta, float* C, int* ldc);
	};
#endif

template<class T>
struct Matrix
{
	int m, n;
#ifdef USE_EIGEN
	Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v;
#else
	std::vector<T> v;
#endif
	
	Matrix(){}
	Matrix( const int& m, const int& n ) :m(m), n(n)
	{
#ifdef USE_EIGEN
		v = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(m, n);
		for( int i = 0; i < m; ++i ) for( int j = 0; j < n; ++j ) v(i,j) = T();
#else
		v = std::vector<T>(m*n, T());
#endif
	}

	Matrix( const std::vector<T>& v ):m(v.size()), n(1)
	{
#ifdef USE_EIGEN
		this->v = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(m, n);
		for( int i = 0; i < m; ++i ) this->v(i, 0) = v[i];
#else
		this->v = std::vector<T>(v.size(), T());
		for( int i = 0; i < m; ++i ) this->v[i] = v[i];
#endif
	}

#ifdef USE_EIGEN
	Matrix( const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A ) :v(A), m(A.rows()), n(A.cols()) {}
#endif
	
	static Matrix<T> eye ( const int& m, const int& n )
	{
		Matrix<T> ret(m, n);
		for( int i = 0; i < std::min(m,n); ++i ) ret(i,i) = 1.0;
		return ret;
	}

	static Matrix<T> ones ( const int& m, const int& n )
	{
		Matrix<T> ret(m, n);
		for( int i = 0; i < m; ++i ) for( int j = 0; j < n; ++j ) ret(i,j) = 1.0;
		return ret;
	}

	static Matrix<T> zeros ( const int& m, const int& n )
	{
		return Matrix<T>(m, n);
	}

	static Matrix<T> transpose( const Matrix<T>& mat )
	{
		int i, j;
		int m = mat.m, n = mat.n;
		Matrix<T> ret(n, m);

#pragma omp parallel for default(none) \
	private(i,j) shared(m,n,mat,ret)
		for( i = 0; i < n; ++i ){
			for( j = 0; j < m; ++j ){
				ret(i,j) = mat(j,i);
			}
		}

		return ret;
	}

	static Matrix<T> hadamard ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;

		int i, j;
		Matrix<T> ret(m, n);
#pragma omp parallel for default(none) \
	private(i,j) shared(m,n,m1,m2,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j )
				ret(i,j) = m1(i,j)*m2(i,j);

		return ret;
	}
	
	static T norm_fro ( const Matrix<T>& mat )
	{
		int m = mat.m, n = mat.n;
		T ret = 0.0;

		for( int i = 0; i < m; ++i )
			for( int j = 0; j < n; ++j )
				ret += mat(i,j)*mat(i,j);

		return sqrt(ret);
	}

	const T& operator () ( int i, int j ) const
	{
#ifdef USE_EIGEN
		return v(i, j);
#else		
		return v[i*n + j];
#endif
	}

	T& operator () ( int i, int j )
	{
#ifdef USE_EIGEN
		return v(i, j);
#else		
		return v[i*n + j];
#endif
	}

	Matrix<T>& operator += ( const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
#ifdef USE_EIGEN
		this->v += m1.v;
#else
		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(m,n,m1)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j )
				(*this)(i,j) += m1(i,j);
#endif

		return *this;
	}

	Matrix<T>& operator -= ( const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
#ifdef USE_EIGEN
		this->v -= m1.v;
#else
		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(m,n,m1)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j )
				(*this)(i,j) -= m1(i,j);
#endif

		return *this;
	}

	Matrix<T>& operator *= ( const Matrix<T>& m1 )
	{
#ifdef USE_EIGEN
		this->v *= m1.v;
#else
		*this = *this*m1;
#endif

		return *this;
	}

	Matrix<T>& operator *= ( const T& c )
	{
#ifdef USE_EIGEN
		this->v *= c;
#else
		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(c)
		for( i = 0; i < this->m; ++i )
			for( j = 0; j < this->n; ++j )
				*this(i,j) *= c;
#endif

		return *this;
	}
	
	Matrix<T>& operator /= ( const T& c )
	{
#ifdef USE_EIGEN
		this->v /= c;
#else
		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(c)
		for( i = 0; i < this->m; ++i )
			for( j = 0; j < this->n; ++j )
				*this(i,j) /= c;
#endif

		return *this;
	}

	friend Matrix<T> operator + ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);

#ifdef USE_EIGEN
		ret.v = m1.v + m2.v;
#else
		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(m,n,m1,m2,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j )
				ret(i,j) = m1(i,j) + m2(i,j);
		
#endif
		return ret;
	}
	
	friend Matrix<T> operator - ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);
#ifdef USE_EIGEN
		ret.v = m1.v - m2.v;
#else
		int i, j;
#pragma omp parallel for default(none)			\
	private(i,j) shared(m,n,m1,m2,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j )
				ret(i,j) = m1(i,j) - m2(i,j);
		
#endif
		return ret;
	}

	friend Matrix<T> operator * ( const Matrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		Matrix<T> ret(m, n);
#ifdef USE_EIGEN
		ret.v = m1.v*m2.v;
#elif USE_BLAS
		T ONE = 1.0, ZERO = 0.0;
		std::vector<T> tmp_m1(m1.m*m1.n), tmp_m2(m2.m*m2.n), tmp_ret(m*n);

		int i, j;
#pragma omp parallel for default(none) \
	private(i,j) shared(tmp_m1, m1)
		for( i = 0; i < m1.m; ++i ) for( j = 0; j < m1.n; ++j ) tmp_m1[i+m1.m*j] = m1(i,j);
#pragma omp parallel for default(none) \
	private(i,j) shared(tmp_m2, m2)
		for( i = 0; i < m2.m; ++i ) for( j = 0; j < m2.n; ++j ) tmp_m2[i+m2.m*j] = m2(i,j);

		int lda = m1.m, ldb = m2.m;
		dgemm_("N", "N", &m, &n, &l, &ONE, &tmp_m1[0], &lda, &tmp_m2[0], &ldb, &ZERO, &tmp_ret[0], &lda);
		// sgemm_("N", "N", &m, &n, &l, &ONE, &tmp_m1[0], &lda, &tmp_m2[0], &ldb, &ZERO, &tmp_ret[0], &lda);

#pragma omp parallel for default(none) \
	private(i,j) shared(ret, tmp_ret, m, n)
		for( i = 0; i < m; ++i ) for( j = 0; j < n; ++j ) ret(i,j) = tmp_ret[i+m*j];
#else
		int i, j, k;
#pragma omp parallel for default(none)	\
	private(i,j,k) shared(m,n,l,m1,m2,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j ){
				T sum = 0.0;
				for( k = 0; k < l; ++k )
					sum += m1(i,k)*m2(k,j);
				ret(i,j) = sum;
			}
		
#endif
		return ret;
	}

	friend Matrix<T> operator * ( const T& c, const Matrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		Matrix<T> ret(m, n);
#ifdef USE_EIGEN
		ret.v = c*m1.v;
#else
		int i, j;

#pragma omp parallel for default(none) \
	private(i,j) shared(m,n,c,m1,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j ){
				ret(i,j) = c*m1(i,j);
			}
		
#endif
		return ret;
	}

	friend Matrix<T> operator * ( const Matrix<T>& m1, const T& c )
	{
		return c*m1;
	}
	 
	friend Matrix<T> operator / ( const Matrix<T>& m1, const T& c )
	{
		return (1.0/c)*m1;
	}

	friend std::ostream& operator << ( std::ostream& os, const Matrix<T>& A )
	{
		for( int i = 0; i < A.m; ++i ){
			for( int j = 0; j < A.n; ++j ){
				if( j != 0 ) os << " ";
				os << std::scientific << std::setprecision(3) << std::setw(10) << A(i,j);
			}
			std::cout << std::endl;
		}
		return os;
	}
};

template<class T>
int pivoting ( const Matrix<T>& A, const Matrix<T>& L, const Matrix<T>& U, const int& j )
{
	int m = A.m, n = A.n;
	double max_pivot = L(j,0);
	int idx = j;

	for( int i = j; i < std::min(m, n); ++i ){
		double sum = 0.0;
		for( int k = 0; k < n; ++k ) max_pivot = std::max(max_pivot, L(i,k));

		for( int k = 0; k < i; ++k ) sum += L(i,k)*U(k,j)/max_pivot;

		double x = A(i,j)/max_pivot + sum;
		if( max_pivot < x ){
			max_pivot = x;
			idx = i;
		}
	}

	return idx;
}

template<class T>
void LU_decomp ( Matrix<T> A, Matrix<T>& L, Matrix<T>& U, Matrix<T>& P )
{
	int m = A.m, n = A.n;
	
	if( m > n ){
		L = Matrix<T>::eye(m, n);
		U = Matrix<T>::zeros(n, n);
	}
	else{
		L = Matrix<T>::eye(m, m);
		U = Matrix<T>::eye(m, n);
	}
	P = Matrix<T>::eye(m, m);

	for( int i = 0; i < m; ++i ){
		int idx = pivoting(A, L, U, i);

		if( idx != i ){
			for( int j = 0; j < std::min(m,n); ++j ) std::swap(A(i,j), A(idx,j));
			P(i,i) = P(idx,idx) = 0.0;
			P(i,idx) = P(idx,i) = 1.0;
		}

		for( int j = 0; j < n; ++j ){
			double sum = 0.0;
			for( int k = 0; k < std::min(i,j); ++k ) sum += L(i,k)*U(k,j);

			if( i > j ) L(i,j) = (A(i,j) - sum)/U(j,j);
			else U(i,j) = A(i,j) - sum;
		}
	}
}

template<class T>
Matrix<T> FBS ( const Matrix<T>& L, const Matrix<T>& U, const Matrix<T>& P, Matrix<T> B )
{
	int m = L.m, n = L.n;
	Matrix<T> Y = Matrix<T>::zeros(B.m, B.n), X = Matrix<T>::zeros(B.m, B.n);

	B = Matrix<T>::transpose(P)*B;
	
	for( int i = 0; i < m; ++i ){
		std::vector<T> sum(B.n, T());
		for( int j = 0; j < B.n; ++j ){
			for( int k = 0; k < i; ++k ){
				sum[j] += L(i,k)*Y(k,j);
			}
		}
		
		for( int j = 0; j < B.n; ++j ){
			Y(i,j) = (B(i,j) - sum[j]) / L(i,i);
		}
	}
	
	for( int i = m-1; i >= 0; --i ){
		std::vector<T> sum(B.n, T());
		for( int j = 0; j < B.n; ++j ){
			for( int k = i; k < n; ++k ){
				sum[j] += U(i,k)*X(k,j);
			}
		}
		
		for( int j = 0; j < B.n; ++j ){
			X(i,j) = (Y(i,j) - sum[j]) / U(i,i);
		}
	}
	
	return X;
}

#endif
