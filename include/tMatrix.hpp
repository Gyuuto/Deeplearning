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

#pragma omp parallel for
		for( int i = 0; i < mat->m; ++i )
			for( int j = 0; j < mat->n; ++j )
				ret(i, j) = (*mat)(j, i);
		
		return ret;
	}
};

#endif
