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
		cl_event event;
		clMatrix<T> ret(m, n);

		clblasSgemm( clblasRowMajor, clblasNoTrans, clblasTrans,
					 m, n, l, 1.0f,
					 m1.v, 0, m1.n, m2.mat->v, 0, m2.m,
					 0.0, ret.v, 0, ret.n,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n*l;

		return ret;
	}

	friend clMatrix<T> operator * ( const cltMatrix<T>& m1, const clMatrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		cl_event event;
		clMatrix<T> ret(m, n);

		cl_int err = clblasSgemm( clblasRowMajor, clblasTrans, clblasNoTrans,
								  m, n, l, 1.0f,
								  m1.mat->v, 0, m1.m, m2.v, 0, m2.n,
								  0.0, ret.v, 0, ret.n,
								  1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n*l;

		return ret;
	}
	
	friend clMatrix<T> operator * ( const cltMatrix<T>& m1, const cltMatrix<T>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		cl_event event;
		clMatrix<T> ret(m, n);

		clblasSgemm( clblasRowMajor, clblasTrans, clblasTrans,
					 m, n, l, 1.0f,
					 m1.mat->v, 0, m1.m, m2.mat->v, 0, m2.m,
					 0.0, ret.v, 0, ret.n,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n*l;

		return ret;
	}
};

#endif
