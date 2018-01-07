#ifndef CLMATRIX_HPP
#define CLMATRIX_HPP

#include <vector>
#include <functional>

#include "Matrix.hpp"

#include <cublas.h>
#include <cuda_runtime_api.h>
#include "CUDA/cuda_kernel.h"

#include <chrono>

template<class T>
struct cudaMatrix;

#include "cudatMatrix.hpp"

template<class T>
struct cudaMatrix
{
	int m, n;
	long long mem_size;
    T* v;
	
	cudaMatrix()
        : m(0), n(0), mem_size(0), v(NULL)
	{}
	
	cudaMatrix( const int& m, const int& n )
        : m(m), n(n), mem_size((long long)m*n)
	{
        cudaMalloc((void**)&v, m*n*sizeof(T));
	}

	cudaMatrix( const std::vector<T>& v )
        : m(v.size()), n(1), mem_size((long long)m*n)
	{
        cudaMalloc((void**)&v, m*sizeof(T));
        cudaMemcpy(v, v.begin(), m*sizeof(T), cudaMemcpyHostToDevice);
	}

	cudaMatrix( const cudaMatrix<T>& mat )
        : m(mat.m), n(mat.n), mem_size((long long)m*n)
	{
        cudaMalloc((void**)&v, m*n*sizeof(T));
        cudaMemcpy(v, mat.v, m*n*sizeof(T), cudaMemcpyDeviceToDevice);
	}

	cudaMatrix( const Matrix<T>& mat )
        : m(mat.m), n(mat.n), mem_size((long long)m*n)
	{
        cudaMalloc((void**)&v, m*n*sizeof(T));
        cudaMemcpy(v, mat.v, m*n*sizeof(T), cudaMemcpyHostToDevice);
	}

	cudaMatrix<T>& operator = ( const cudaMatrix<T>& mat )
	{
        if( this->v != NULL ) cudaFree(this->v);
        this->m = mat.m; this->n = mat.n;

        cudaMalloc((void**)&v, m*n*sizeof(T));
        cudaMemcpy(v, mat.v, m*n*sizeof(T), cudaMemcpyDeviceToDevice);

        return *this;
	}

	cudaMatrix<T>& operator = ( const Matrix<T>& mat )
	{
        if( this->v != NULL ) cudaFree(this->v);
        this->m = mat.m; this->n = mat.n;

        cudaMalloc((void**)&v, m*n*sizeof(T));
        cudaMemcpy(v, mat.v, m*n*sizeof(T), cudaMemcpyHostToDevice);

        return *this;
	}

	~cudaMatrix ()
	{
        if( this->v != NULL ) cudaFree(this->v);
	}
	
	static cudaMatrix<T> eye ( const int& m, const int& n )
	{
        cudaMatrix<T> ret(m, n);
        
        cuda_eye_kernel( m, n, ret.v );

        return ret;
	}

	static cudaMatrix<T> ones ( const int& m, const int& n )
	{
        cudaMatrix<T> ret(m, n);

        cuda_ones_kernel( m, n, ret.v );

        return ret;
	}

	static cudaMatrix<T> zeros ( const int& m, const int& n )
	{
        cudaMatrix<T> ret(m, n);

        cuda_zeros_kernel( m, n, ret.v );

        return ret;
	}
	static cudaMatrix<T> zeros ( cudaMatrix<T>& x )
	{
        cuda_zeros_kernel( x.m, x.n, x.v );
    }

	static cudatMatrix<T> transpose( const cudaMatrix<T>& mat )
	{
        return cudatMatrix<T>(&mat);
	}

	static cudaMatrix<T> hadamard ( const cudaMatrix<T>& m1, const cudaMatrix<T>& m2 )
	{
        assert( m1.m == m2.m && m1.n == m2.n );

        cudaMatrix<T> ret(m1.m, m1.n);

        cuda_hadamard_kernel( ret.m, ret.n, m1.v, m2.v, ret.v );

        return ret;
	}

	static T norm_fro ( const cudaMatrix<T>& mat )
	{
        return cublasSdot ( mat.m*mat.n, mat.v, 1, mat.v, 1 );
	}

	static cudaMatrix<T> to_matrix ( const std::vector<cudaMatrix<T>>& tensor )
	{
        int num_map = tensor.size();
        cudaMatrix<T> ret(num_map*tensor[0].m, tensor[0].n);

        for( int i = 0; i < num_map; ++i ) {
            ret.sub(i*tensor[i].m, 0, tensor[i].m, tensor[i].n, tensor[i]);
        }

        return ret;
	}

	static std::vector<cudaMatrix<T>> to_tensor ( const cudaMatrix<T>& mat, int num_map )
	{
        std::vector<cudaMatrix<T>> ret(num_map);

        int leng = mat.m / num_map;
        for( int i = 0; i < num_map; ++i )
            ret[i] = mat.sub(i*leng, 0, leng, mat.n);

        return ret;
	}		

	
	T get_element ( int i, int j ) const
	{
        assert( 0 <= i && i < this->m );
        assert( 0 <= j && j < this->n );

        T ret;
        cudaMemcpy(&ret, this->v + (i * this->n + j)*sizeof(T), 1*sizeof(T), cudaMemcpyDeviceToHost);

        return ret;
	}

	void set_element ( int i, int j, T val )
	{
        assert( 0 <= i && i < this->m );
        assert( 0 <= j && j < this->n );

        cudaMemcpy(this->v + (i * this->n + j)*sizeof(T), &val, 1*sizeof(T), cudaMemcpyHostToDevice);
	}

	cudaMatrix<T>& operator += ( const cudaMatrix<T>& m1 )
	{
        int m = m1.m, n = m1.n;

        assert( this->m == m1.m && this->n == m1.n );

        cublasSaxpy(m*n, 1.0f, m1.v, 1, this->v, 1);

        cnt_flop += (long long)m*n;

        return *this;
	}

	cudaMatrix<T>& operator -= ( const cudaMatrix<T>& m1 )
	{
        int m = m1.m, n = m1.n;

        assert( this->m == m1.m && this->n == m1.n );

        cublasSaxpy(m*n, -1.0f, m1.v, 1, this->v, 1);

        cnt_flop += (long long)m*n;

        return *this;        
	}

	cudaMatrix<T>& operator *= ( const cudaMatrix<T>& m1 )
	{
        *this = *this * m1;

        return *this;
	}

	cudaMatrix<T>& operator *= ( const cudatMatrix<T>& m1 )
	{
        *this = *this * m1;
	}

	cudaMatrix<T>& operator *= ( const T& c )
	{
        int m = this->m, n = this->n;

        cublasSaxpy(m*n, c - 1.0, this->v, 1, this->v, 1);

        cnt_flop += (long long)m*n;

        return *this;
	}
	
	cudaMatrix<T>& operator /= ( const T& c )
	{
        int m = this->m, n = this->n;

        cublasSaxpy(m*n, 1.0/c - 1.0, this->v, 1, this->v, 1);

        cnt_flop += (long long)m*n;

        return *this;
	}

	friend cudaMatrix<T> operator + ( const cudaMatrix<T>& m1, const cudaMatrix<T>& m2 )
	{
        assert( m1.m == m2.m && m1.n == m2.n );

        int m = m1.m, n = m2.n;
        cudaMatrix<T> ret = m1;

        cublasSaxpy( m*n, 1.0f, m2.v, 1, ret.v, 1 );

        cnt_flop += (long long)m*n;

        return ret;
	}
	
	friend cudaMatrix<T> operator - ( const cudaMatrix<T>& m1, const cudaMatrix<T>& m2 )
	{
        assert( m1.m == m2.m && m1.n == m2.n );

        int m = m1.m, n = m2.n;
        cudaMatrix<T> ret = m1;

        cublasSaxpy( m*n, -1.0f, m2.v, 1, ret.v, 1 );

        cnt_flop += (long long)m*n;

        return ret;        
	}

	friend cudaMatrix<T> operator - ( const cudaMatrix<T>& m1 )
	{
        return -1*m1;
	}

	friend cudaMatrix<T> operator * ( const cudaMatrix<T>& m1, const cudaMatrix<T>& m2 )
	{
        assert( m1.n == m2.m );

        int m = m1.m, n = m2.n, l = m1.n;
        cudaMatrix<T> ret(m, n);

        cublasSgemm('N', 'N', n, m, l, 1.0f, m2.v, n, m1.v, l, 0.0f, ret.v, n);

		cnt_flop += (long long)m*n*(2*l-1);

        return ret;
	}

	friend cudaMatrix<T> operator * ( const T& c, const cudaMatrix<T>& m1 )
	{
        int m = m1.m, n = m1.n;

        cudaMatrix<T> ret = m1;

        cublasSaxpy( m*n, c - 1.0, m1.v, 1, ret.v, 1 );

        cnt_flop += (long long)m*n;

        return ret;
	}

	friend cudaMatrix<T> operator * ( const cudaMatrix<T>& m1, const T& c )
	{
        return c*m1;
	}
	 
	friend cudaMatrix<T> operator / ( const cudaMatrix<T>& m1, const T& c )
	{
        return (1.0/c)*m1;
	}

	cudaMatrix<T> sub ( int y, int x, int h, int w ) const
	{
        assert( 0 <= y && y < this->m );
        assert( 0 <= x && x < this->n );
        assert( y < y+h && y+h <= this->m );
        assert( x < x+w && x+w <= this->n );

        cudaMatrix<T> ret(h, w);

        cuda_sub_kernel( y, x, h, w, this->v, this->n, ret.v, ret.n );

        return ret;
	}

	void sub ( int y, int x, int h, int w, const cudaMatrix<T>& mat )
	{
        assert( 0 <= y && y < this->m );
        assert( 0 <= x && x < this->n );
        assert( 0 < h && h <= mat.m );
        assert( 0 < w && w <= mat.m );
        assert( y+h <= this->m );
        assert( x+w <= this->n );

        cuda_sub_kernel( y, x, h, w, mat.v, mat.n, this->v, this->n );
	}

	Matrix<T> get_matrix ( ) const
	{
        Matrix<T> ret(this->m, this->n);

        cudaMemcpy( ret.v, this->v, this->m*this->n*sizeof(T), cudaMemcpyDeviceToHost );

        return ret;
	}

	void clip( const T& val )
	{
        cuda_clip_kernel ( val, this->v, this->m, this->n );
	}

	void mult ( const T& alpha, const cudaMatrix<T>& B, const T& beta, cudaMatrix<T>& C ) const
	{
        int m = this->m, n = B.n, l = this->n;

        assert( this->n == B.m );
        assert( m == C.m );
        assert( C.n == n );
        
        cublasSgemm( 'N', 'N', n, m, l, alpha, B.v, n, this->v, l, beta, C.v, n );

        cnt_flop += (long long)m*n*(2*l-1);
	}

	void mult ( const T& alpha, const cudatMatrix<T>& B, const T& beta, cudaMatrix<T>& C ) const
	{
        int m = this->m, n = B.n, l = this->n, k = B.m;

        assert( this->n == B.m );
        assert( m == C.m );
        assert( n == C.n );
        
        cublasSgemm( 'T', 'N', n, m, l, alpha, B.mat->v, k, this->v, l, beta, C.v, n );

        cnt_flop += (long long)m*n*(2*l-1);
	}

	void hadamard ( const cudaMatrix<T>& A )
	{
        assert( this->m == A.m && this->n == A.n );
        
        cuda_hadamard_inplace_kernel( this->m, this->n, this->v, A.v );
	}

	void reshape ( int m, int n )
	{
        assert( (long long)m*n <= mem_size );

        if( this->m == m && this->n == n ) return;

        this->m = m; this->n = n;
	}

	void copy ( const cudaMatrix<T>& A )
	{
        cudaMemcpy(this->v, A.v, this->m*this->n*sizeof(T), cudaMemcpyDeviceToDevice);
    }
};

#endif
