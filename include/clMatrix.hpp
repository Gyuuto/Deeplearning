#ifndef CLMATRIX_HPP
#define CLMATRIX_HPP

#include <vector>
#include <functional>
#include <clBLAS.h>

#include "clDeviceManager.hpp"
#include "Matrix.hpp"

template<class T>
struct clMatrix;

#include "cltMatrix.hpp"

template<class T>
struct clMatrix
{
	int m, n;
	int mem_size;
	// std::vector<T> v;
	cl_mem M, N, v;
	
	clMatrix(): m(0), n(0), v(NULL), mem_size(0)
	{
		cl_int err;
		M = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
									sizeof(int), &m, 0, NULL, NULL );

		N = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
									sizeof(int), &n, 0, NULL, NULL );
	}
	
	clMatrix( const int& m, const int& n ) :m(m), n(n)
	{
		cl_int err;
		M = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
									sizeof(int), &m, 0, NULL, NULL );

		N = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
									sizeof(int), &n, 0, NULL, NULL );

		v = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, m*n*sizeof(T), NULL, &err);
		mem_size = m*n;
	}

	clMatrix( const std::vector<T>& v ):m(v.size()), n(1)
	{
		cl_int err;

		M = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
									sizeof(int), &m, 0, NULL, NULL );

		N = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
									sizeof(int), &n, 0, NULL, NULL );

		this->v = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, m*n*sizeof(T), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->v, CL_TRUE, 0,
									m*sizeof(T), &v[0], 0, NULL, NULL );
		mem_size = m*n;
	}

	clMatrix( const clMatrix<T>& mat )
	{
		m = mat.m; n = mat.n;
		mem_size = m*n;
		if( m == 0 || n == 0 ){
			v = NULL;
		}
		else{
			cl_int err;
			M = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
										sizeof(int), &m, 0, NULL, NULL );

			N = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
										sizeof(int), &n, 0, NULL, NULL );

			v = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, m*n*sizeof(T), NULL, &err);
			err = clEnqueueCopyBuffer( cl_device_manager.get_queue(), mat.v, v, 0, 0, m*n*sizeof(T), 0, NULL, NULL );
		}
	}

	clMatrix( const Matrix<T>& mat )
	{
		m = mat.m; n = mat.n;
		mem_size = m*n;
		if( m == 0 || n == 0 ){
			v = NULL;
		}
		else{
			cl_int err;
			M = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
										sizeof(int), &m, 0, NULL, NULL );

			N = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
										sizeof(int), &n, 0, NULL, NULL );

			v = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, m*n*sizeof(T), NULL, &err);
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), v, CL_TRUE, 0,
										m*n*sizeof(T), mat.v, 0, NULL, NULL );
		}
	}	 

	clMatrix<T>& operator = ( const clMatrix<T>& mat )
	{
		cl_int err;
		if( mat.m == 0 || mat.n == 0 ){
			if( mem_size != 0 ){
				clReleaseMemObject(v);
			}

			m = mat.m; n = mat.n;
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
										sizeof(int), &m, 0, NULL, NULL );
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
										sizeof(int), &n, 0, NULL, NULL );
			v = NULL;
			
			return *this;
		}

		if( mat.m*mat.n > mem_size ) {
			clReleaseMemObject(v);
			v = NULL;

			mem_size = mat.m*mat.n;
		}
		if( !(m == mat.m && n == mat.n) ){
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
										sizeof(int), &mat.m, 0, NULL, NULL );
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
										sizeof(int), &mat.n, 0, NULL, NULL );
		}

		m = mat.m; n = mat.n;
		if( v == NULL ){
			v = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, m*n*sizeof(T), NULL, &err);
		}
		err = clEnqueueCopyBuffer( cl_device_manager.get_queue(), mat.v, v, 0, 0, m*n*sizeof(T), 0, NULL, NULL );

		return *this;
	}

	clMatrix<T>& operator = ( const Matrix<T>& mat )
	{
		cl_int err;
		if( mat.m == 0 || mat.n == 0 ){
			if( mem_size != 0 ){
				clReleaseMemObject(v);
			}

			m = mat.m; n = mat.n;
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
										sizeof(int), &m, 0, NULL, NULL );
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
										sizeof(int), &n, 0, NULL, NULL );
			v = NULL;
			
			return *this;
		}

		if( mat.m*mat.n > mem_size ) {
			clReleaseMemObject(v);
			v = NULL;

			mem_size = mat.m*mat.n;
		}
		if( !(m == mat.m && n == mat.n) ){
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->M, CL_TRUE, 0,
										sizeof(int), &mat.m, 0, NULL, NULL );
			err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->N, CL_TRUE, 0,
										sizeof(int), &mat.n, 0, NULL, NULL );
		}

		m = mat.m; n = mat.n;
		if( v == NULL ){
			v = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, m*n*sizeof(T), NULL, &err);
		}
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), v, CL_TRUE, 0,
									m*n*sizeof(T), mat.v, 0, NULL, NULL );

		return *this;
	}

	~clMatrix ()
	{
		if( v != NULL ){
			clReleaseMemObject( M );
			clReleaseMemObject( N );
			clReleaseMemObject( v );
		}
	}
	
	static clMatrix<T> eye ( const int& m, const int& n )
	{
		clMatrix<T> ret(m, n);

		cl_device_manager.set_argument( PRG::CLMAT_ZEROS, 0, &ret.v );
		cl_device_manager.run_kernel( PRG::CLMAT_ZEROS, ret.m*ret.n, 1 );

		cl_device_manager.set_argument( PRG::CLMAT_EYE, 0, &ret.v );
		cl_device_manager.set_argument( PRG::CLMAT_EYE, 1, &ret.M );
		cl_device_manager.set_argument( PRG::CLMAT_EYE, 2, &ret.N );
		cl_device_manager.run_kernel( PRG::CLMAT_EYE, std::min(ret.m, ret.n), 1 );

		return ret;
	}

	static clMatrix<T> ones ( const int& m, const int& n )
	{
		clMatrix<T> ret(m, n);

		cl_device_manager.set_argument( PRG::CLMAT_ONES, 0, &ret.v );
		cl_device_manager.run_kernel( PRG::CLMAT_ONES, ret.m*ret.n, 1 );

		return ret;
	}

	static clMatrix<T> zeros ( const int& m, const int& n )
	{
		clMatrix<T> ret(m, n);

		cl_device_manager.set_argument( PRG::CLMAT_ZEROS, 0, &ret.v );
		cl_device_manager.run_kernel( PRG::CLMAT_ZEROS, ret.m*ret.n, 1 );

		return ret;
	}

	static cltMatrix<T> transpose( const clMatrix<T>& mat )
	{
		return cltMatrix<T>(&mat);
	}

	static clMatrix<T> hadamard ( const clMatrix<T>& m1, const clMatrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		clMatrix<T> ret(m, n);

		cl_device_manager.set_argument( PRG::CLMAT_HADAMARD, 0, &ret.v );
		cl_device_manager.set_argument( PRG::CLMAT_HADAMARD, 1, &m1.v );
		cl_device_manager.set_argument( PRG::CLMAT_HADAMARD, 2, &m2.v );
		cl_device_manager.run_kernel( PRG::CLMAT_HADAMARD, ret.m*ret.n, 1 );

		return ret;
	}
	
	static T norm_fro ( const clMatrix<T>& mat )
	{
		int m = mat.m, n = mat.n;
		cl_int err;
		cl_event event;

		cl_mem buf = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, sizeof(float), NULL, &err );
		cl_mem scratch_buf = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, m*n*sizeof(float), NULL, &err );
				
		clblasSdot( m*n, buf, 0, mat.v, 0, 1, mat.v, 0, 1, scratch_buf,
					1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		T ret;
		clEnqueueReadBuffer( cl_device_manager.get_queue(), buf, CL_TRUE, 0, sizeof(T), &ret, 0, NULL, NULL );
		
		return sqrt(ret);
	}

	T get_element ( int i, int j ) const
	{
		T ret;
		clEnqueueReadBuffer( cl_device_manager.get_queue(), v, CL_TRUE, (i*n + j)*sizeof(T),
							 sizeof(T), &ret, 0, NULL, NULL );
		return ret;
	}

	void set_element ( int i, int j, T val )
	{
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), v, CL_TRUE, (i*n + j)*sizeof(T),
							  sizeof(T), &val, 0, NULL, NULL );
	}

	clMatrix<T>& operator += ( const clMatrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		cl_event event;

		clblasSaxpy( m*n, 1.0f,
					 m1.v, 0, 1, this->v, 0, 1,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );
		
		cnt_flop += (long long)m*n;

		return *this;
	}

	clMatrix<T>& operator -= ( const clMatrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		cl_event event;

		clblasSaxpy( m*n, -1.0f,
					 m1.v, 0, 1, this->v, 0, 1,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n;

		return *this;
	}

	clMatrix<T>& operator *= ( const clMatrix<T>& m1 )
	{
		*this = *this*m1;

		return *this;
	}

	clMatrix<T>& operator *= ( const cltMatrix<T>& m1 )
	{
		*this = *this*m1;

		return *this;
	}

	clMatrix<T>& operator *= ( const T& c )
	{
		int m = this->m, n = this->n;
		cl_event event;

		clblasSaxpy( m*n, c - 1.0,
					 this->v, 0, 1, this->v, 0, 1,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n;

		return *this;
	}
	
	clMatrix<T>& operator /= ( const T& c )
	{
		int m = this->m, n = this->n;
		cl_event event;

		clblasSaxpy( m*n, 1.0/c - 1.0,
					 this->v, 0, 1, this->v, 0, 1,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n;

		return *this;
	}

	friend clMatrix<T> operator + ( clMatrix<T>& m1, clMatrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		clMatrix<T> ret = m1;
		cl_int err;
		cl_event event;

		clblasSaxpy( m*n, 1.0f,
					 m2.v, 0, 1, ret.v, 0, 1,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n;

		return ret;
	}
	
	friend clMatrix<T> operator - ( const clMatrix<T>& m1, const clMatrix<T>& m2 )
	{
		int m = m1.m, n = m1.n;
		clMatrix<T> ret = m1;
		cl_event event;

		clblasSaxpy( m*n, -1.0f,
					 m2.v, 0, 1, ret.v, 0, 1,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n;

		return ret;
	}

	friend clMatrix<float> operator * ( const clMatrix<float>& m1, const clMatrix<float>& m2 )
	{
		int m = m1.m, n = m2.n, l = m1.n;
		cl_event event;
		clMatrix<float> ret(m, n);

		clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
					 m, n, l, 1.0f,
					 m1.v, 0, m1.n, m2.v, 0, m2.n,
					 0.0, ret.v, 0, ret.n,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n*l;

		return ret;
	}

	friend clMatrix<T> operator * ( const T& c, const clMatrix<T>& m1 )
	{
		int m = m1.m, n = m1.n;
		cl_event event;

		clMatrix<T> ret = m1;
		
		clblasSaxpy( m*n, c - 1.0,
					 ret.v, 0, 1, ret.v, 0, 1,
					 1, cl_device_manager.get_queue_ptr(), 0, NULL, &event );
		clWaitForEvents( 1, &event );

		cnt_flop += (long long)m*n;

		return ret;
	}

	friend clMatrix<T> operator * ( const clMatrix<T>& m1, const T& c )
	{
		return c*m1;
	}
	 
	friend clMatrix<T> operator / ( const clMatrix<T>& m1, const T& c )
	{
		return (1.0f/c)*m1;
	}

	clMatrix<T> sub ( int y, int x, int h, int w ) const
	{
		clMatrix<T> ret(h, w);
		cl_int err;
		cl_mem buf_x, buf_y;

		buf_x = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), buf_x, CL_TRUE, 0,
									sizeof(int), &x, 0, NULL, NULL );
		buf_y = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), buf_y, CL_TRUE, 0,
									sizeof(int), &y, 0, NULL, NULL );

		cl_device_manager.set_argument( PRG::CLMAT_SUB, 0, &ret.v );
		cl_device_manager.set_argument( PRG::CLMAT_SUB, 1, &v );
		cl_device_manager.set_argument( PRG::CLMAT_SUB, 2, &buf_x );
		cl_device_manager.set_argument( PRG::CLMAT_SUB, 3, &buf_y );
		cl_device_manager.set_argument( PRG::CLMAT_SUB, 4, &N );
		
		cl_device_manager.run_kernel( PRG::CLMAT_SUB, ret.m, ret.n );
		
		return ret;
	}

	Matrix<T> get_matrix ( ) const
	{
		cl_int err;
		Matrix<T> ret(this->m, this->n);

		err = clEnqueueReadBuffer( cl_device_manager.get_queue(), this->v, CL_TRUE, 0,
								   this->m*this->n*sizeof(T), ret.v, 0, NULL, NULL );
		return ret;
	}
};

#endif
