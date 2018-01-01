#ifndef CLTMATRIX_HPP
#define CLTMATRIX_HPP

template<class T>
struct cltMatrix
{
	int m, n;
	const clMatrix<T>* mat;

	static clMatrix<T> transpose( const cltMatrix<T>& mat )
	{
		return *(mat.mat);
	}

	cltMatrix( const clMatrix<T>* mat )
	{
		this->mat = mat;
		m = mat->n; n = mat->m;
	}
	
	clMatrix<T> inplace ()
	{
		clMatrix<T> ret(mat->n, mat->m);

		// do it using OpenCL
		
		return ret;
	}
	
	friend clMatrix<T> operator * ( const clMatrix<T>& m1, const cltMatrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		clMatrix<T> ret(m, n);

		clblasSgemm( clblasRowMajor, clblasNoTrans, clblasTrans,
					 m, n, l, 1.0f,
					 m1.v, 0, m1.n, m2.mat->v, 0, m2.m,
					 0.0, ret.v, 0, ret.n,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, NULL );

		cnt_flop += (long long)m*n*(2*l-1);

		return ret;
	}

	friend clMatrix<T> operator * ( const cltMatrix<T>& m1, const clMatrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		clMatrix<T> ret(m, n);

		cl_int err = clblasSgemm( clblasRowMajor, clblasTrans, clblasNoTrans,
								  m, n, l, 1.0f,
								  m1.mat->v, 0, m1.m, m2.v, 0, m2.n,
								  0.0, ret.v, 0, ret.n,
								  1, cl_device_manager.get_queue_ptr(), 0, NULL, NULL );
        if( err != 0 ) printf("WARNING : clblasSgemm failed in cltMatrix::operator*\n");

		cnt_flop += (long long)m*n*(2*l-1);

		return ret;
	}
	
	friend clMatrix<T> operator * ( const cltMatrix<T>& m1, const cltMatrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		clMatrix<T> ret(m, n);

		cl_int err = clblasSgemm( clblasRowMajor, clblasTrans, clblasTrans,
                                  m, n, l, 1.0f,
                                  m1.mat->v, 0, m1.m, m2.mat->v, 0, m2.m,
                                  0.0, ret.v, 0, ret.n,
                                  1, cl_device_manager.get_queue_ptr(), 0, NULL, NULL );
        if( err != 0 ) printf("WARNING : clblasSgemm failed in cltMatrix::operator*\n");

		cnt_flop += (long long)m*n*(2*l-1);

		return ret;
	}

	void mult ( const T& alpha, const clMatrix<T>& B, const T& beta, clMatrix<T>& C ) const
	{
		int m = this->m, n = B.n, l = this->n;

		assert( this->n == B.m );
		assert( m == C.m );
		assert( n == C.n );

		cl_int err = clblasSgemm( clblasRowMajor, clblasTrans, clblasNoTrans,
								  m, n, l,
								  alpha, this->mat->v, 0, this->m, B.v, 0, B.n,
								  beta, C.v, 0, C.n,
								  1, cl_device_manager.get_queue_ptr(), 0, NULL, NULL );
        if( err != 0 ) printf("WARNING : clblasSgemm failed in cltMatrix::mult\n");
		cnt_flop += (long long)m*n*(2*l-1);
	}

	void mult ( const T& alpha, const cltMatrix<T>& B, const T& beta, clMatrix<T>& C ) const
	{
		int m = this->m, n = B.n, l = this->n;

		assert( this->n == B.m );
		assert( m == C.m );
		assert( n == C.n );

		cl_int err = clblasSgemm( clblasRowMajor, clblasTrans, clblasTrans,
								  m, n, l,
								  alpha, this->mat->v, 0, this->m, B.mat->v, 0, B.m,
								  beta, C.v, 0, C.n,
								  1, cl_device_manager.get_queue_ptr(), 0, NULL, NULL );
        if( err != 0 ) printf("WARNING : clblasSgemm failed in cltMatrix::mult\n");
		
		cnt_flop += (long long)m*n*(2*l-1);
	}
};

#endif
