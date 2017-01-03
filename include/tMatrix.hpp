#ifndef TMATRIX_HPP
#define TMATRIX_HPP

template<class T>
struct tMatrix
{
	int m, n;
	const Matrix<T>* mat;

	static Matrix<T> transpose( const tMatrix<T>& mat )
	{
		return *(mat.mat);
	}

	tMatrix( const Matrix<T>* mat )
	{
		this->mat = mat;
		m = mat->n; n = mat->m;
	}
	
	const T& operator () ( int i, int j ) const
	{
		return (*mat)(j, i);
	}

	Matrix<T> inplace ()
	{
		Matrix<T> ret(mat->n, mat->m);

#pragma omp parallel for schedule(auto)
		for( int i = 0; i < mat->m; ++i )
			for( int j = 0; j < mat->n; ++j )
				ret(i, j) = (*mat)(j, i);
		
		return ret;
	}
	
	friend Matrix<T> operator * ( const Matrix<T>& m1, const tMatrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n, k = m2.m;
		Matrix<T> ret(m, n);
#ifdef USE_EIGEN
		ret.v = m1.v*m2.v;
#elif USE_BLAS
		T ONE = 1.0, ZERO = 0.0;
		
		if( m != 0 && n != 0 && l != 0 ){
			dgemm_("T", "N", &n, &m, &l, &ONE,
				   &m2(0,0), &k, &m1(0,0), &l,
				   &ZERO, &ret(0,0), &n);
		}
#else
		int i, j, k;
#pragma omp parallel for default(none)			\
	private(i,j,k) shared(m,n,l,m1,m2,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j ){
				double sum = 0.0;
				for( k = 0; k < l; ++k )
					sum += m1(i,k)*m2(k,j);
				ret(i,j) = sum;
			}
		
#endif
		cnt_flop += m*n*l;

		return ret;
	}

	friend Matrix<T> operator * ( const tMatrix<T>& m1, const Matrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n, k = m2.m;
		Matrix<T> ret(m, n);
#ifdef USE_EIGEN
		ret.v = m1.v*m2.v;
#elif USE_BLAS
		T ONE = 1.0, ZERO = 0.0;

		if( m != 0 && n != 0 && l != 0 ){
			dgemm_("N", "T", &n, &m, &l, &ONE,
				   &m2(0,0), &n, &m1(0,0), &m,
				   &ZERO, &ret(0,0), &n);
			// sgemm_("N", "N", &n, &m, &l, &ONE,
			// 	   &m2(0,0), &n, &m1(0,0), &l,
			// 	   &ZERO, &ret(0,0), &n);
		}
#else
		int i, j, k;
#pragma omp parallel for default(none)	\
	private(i,j,k) shared(m,n,l,m1,m2,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j ){
				double sum = 0.0;
				for( k = 0; k < l; ++k )
					sum += m1(i,k)*m2(k,j);
				ret(i,j) = sum;
			}
		
#endif
		cnt_flop += m*n*l;

		return ret;
	}
	
	friend Matrix<T> operator * ( const tMatrix<T>& m1, const tMatrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		Matrix<T> ret(m, n);
#ifdef USE_EIGEN
		ret.v = m1.v*m2.v;
#elif USE_BLAS
		T ONE = 1.0, ZERO = 0.0;
		
		if( m != 0 && n != 0 && l != 0 ){
			dgemm_("T", "T", &m, &n, &l, &ONE,
				   &m1(0,0), &l, &m2(0,0), &n,
				   &ZERO, &ret(0,0), &m);
			// sgemm_("N", "N", &n, &m, &l, &ONE,
			// 	   &m2(0,0), &n, &m1(0,0), &l,
			// 	   &ZERO, &ret(0,0), &n);
		}
#else
		int i, j, k;
#pragma omp parallel for default(none)	\
	private(i,j,k) shared(m,n,l,m1,m2,ret)
		for( i = 0; i < m; ++i )
			for( j = 0; j < n; ++j ){
				double sum = 0.0;
				for( k = 0; k < l; ++k )
					sum += m1(i,k)*m2(k,j);
				ret(i,j) = sum;
			}
		
#endif
		cnt_flop += m*n*l;

		return ret;
	}
};


#endif
