#ifndef CUDATMATRIX_HPP
#define CUDATMATRIX_HPP

template<class T>
struct cudatMatrix
{
	int m, n;
	const cudaMatrix<T>* mat;

	static cudaMatrix<T> transpose( const cudatMatrix<T>& mat )
	{
        return *(mat.mat);
	}

	cudatMatrix( const cudaMatrix<T>* mat )
	{
        this->mat = mat;
        m = mat->n; n = mat->m;
	}
	
	cudaMatrix<T> inplace ()
	{
        cudaMatrix<T> ret(mat->n, mat->m);

        // do it using kernel

        return ret;
	}
	
	friend cudaMatrix<T> operator * ( const cudaMatrix<T>& m1, const cudatMatrix<T>& m2 )
	{
        int m = m1.m, n = m2.n, l = m1.n;
        int k = m2.m;
        cudaMatrix<T> ret(m, n);

        cublasSgemm('T', 'N', n, m, l, 1.0f, m2.mat->v, k, m1.v, l, 0.0f, ret.v, n);

        cnt_flop += (long long)m*n*(2*l-1);

        return ret;
	}

	friend cudaMatrix<T> operator * ( const cudatMatrix<T>& m1, const cudaMatrix<T>& m2 )
	{
        int m = m1.m, n = m2.n, l = m1.n;
        cudaMatrix<T> ret(m, n);

        cublasSgemm('N', 'T', n, m, l, 1.0f, m2.v, n, m1.mat->v, m, 0.0f, ret.v, n);

        cnt_flop += (long long)m*n*(2*l-1);

        return ret;
	}
	
	friend cudaMatrix<T> operator * ( const cudatMatrix<T>& m1, const cudatMatrix<T>& m2 )
	{
        int m = m1.m, n = m2.n, l = m1.n;
        cudaMatrix<T> ret(m, n);

        cublasSgemm('T', 'T', m, n, l, 1.0f, m2.mat->v, n, m1.mat->v, l, 0.0f, ret.v, m);

        cnt_flop += (long long)m*n*(2*l-1);

        return ret;
	}

	void mult ( const T& alpha, const cudaMatrix<T>& B, const T& beta, cudaMatrix<T>& C ) const
	{
        int m = this->m, n = B.n, l = this->n;

        assert( this->n == B.m );
        assert( m == C.m );
        assert( n == C.n );

        cublasSgemm( 'N', 'T', n, m, l, alpha, B.v, n, this->mat->v, m, beta, C.v, n );

        cnt_flop += (long long)m*n*(2*l-1);
	}

	void mult ( const T& alpha, const cudatMatrix<T>& B, const T& beta, cudaMatrix<T>& C ) const
	{
        int m = this->m, n = B.n, l = this->n;

        assert( this->n == B.m );
        assert( m == C.m );
        assert( n == C.n );

        cublasSgemm( 'T', 'T', m, n, l, alpha, B.mat->v, n, this->mat->v, l, beta, C.v, m );

        cnt_flop += (long long)m*n*(2*l-1);
	}
};

#endif
