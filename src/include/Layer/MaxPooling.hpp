#ifndef MAXPOOLING_HPP
#define MAXPOOLING_HPP

#include "Layer.hpp"

template<template<typename> class Mat, typename Real>
class MaxPooling : public Layer<Mat, Real>
{
private:
	int prev_ldu, ldu;
	int m, n, stride, pad;

#ifdef USE_GPU
	cl_mem cl_num_unit, cl_prev_num_unit;
	cl_mem cl_ldu, cl_prev_ldu;
	cl_mem cl_stride, cl_pad, cl_m, cl_n;
#endif
	
	Mat<Real> S;
public:
	MaxPooling( int prev_num_map, int prev_num_unit, int prev_ldu,
			 int num_map, int num_unit, int ldu,
			 int m, int n, int stride, 
			 const std::shared_ptr<Function<Real>>& f );
	~MaxPooling();

#ifdef USE_MPI
	void init( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world );
#else
	void init( std::mt19937& m );
#endif
	void finalize();

	void calc_gradient ( const Mat<Real>& U_apply, const Mat<Real>& U_diff, const Mat<Real>& delta, std::vector<Mat<Real>>& nabla_W, std::vector<Mat<Real>>& nabla_b );

	void calc_delta ( const Matrix<Real>& U_apply, const Matrix<Real>& U_diff, const Matrix<Real>& delta, Matrix<Real>& nx_delta );
#ifdef USE_GPU
	void calc_delta ( const clMatrix<Real>& U_apply, const clMatrix<Real>& U_diff, const clMatrix<Real>& delta, clMatrix<Real>& nx_delta );
#endif

	Matrix<Real> apply ( const Matrix<Real>& U, bool use_func = true );
	void apply ( const Matrix<Real>& U, Matrix<Real>& ret, bool use_func = true );
#ifdef USE_GPU
	clMatrix<Real> apply ( const clMatrix<Real>& U, bool use_func = true );
	void apply ( const clMatrix<Real>& U, clMatrix<Real>& ret, bool use_func = true );
#endif

	void update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db );
	// Mat unpooling ( const Mat& U );

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename );

#ifdef USE_MPI
	void param_mix ();
#endif
};

template<template<typename> class Mat, typename Real>
MaxPooling<Mat, Real>::MaxPooling( int prev_num_map, int prev_num_unit, int prev_ldu,
							 int num_map, int num_unit, int ldu,
							 int m, int n, int stride, 
							 const std::shared_ptr<Function<Real>>& f )
{
	this->layer_name = "MaxPooling";

	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->prev_ldu = prev_ldu;

	this->num_map = num_map;
	this->num_unit = num_unit;
	this->ldu = ldu;

	this->t_apply = this->t_delta = this->t_grad = 0.0;
	this->t_apply_init = this->t_apply_gemm = this->t_apply_repl = this->t_apply_comm = 0.0;
	this->t_delta_init = this->t_delta_gemm = this->t_delta_repl = this->t_delta_comm = 0.0;
	this->t_grad_init = this->t_grad_gemm = this->t_grad_repl = this->t_grad_comm = 0.0;

	this->m = m; this->n = n; this->stride = stride; this->pad = 0;

	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	if( num_unit%ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of output on MaxPooling layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( prev_num_unit%prev_ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of input on MaxPooling layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( ldu != (prev_ldu + 2*pad - n)/stride + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image width on MaxPooling layer.\n");
			printf("          Estimate width = %d.\n", (prev_ldu + 2*pad - n)/stride + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( num_unit/ldu != (prev_num_unit/prev_ldu + 2*pad - m)/stride + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image height on MaxPooling layer.\n");
			printf("          Estimate height = %d.\n", (prev_num_unit/prev_ldu + 2*pad - m)/stride + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}

	this->func = f;

#ifdef USE_GPU
	cl_int err;
	cl_num_unit = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_num_unit, CL_TRUE, 0,
								sizeof(int), &this->num_unit, 0, NULL, NULL );
	cl_prev_num_unit = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_prev_num_unit, CL_TRUE, 0,
								sizeof(int), &this->prev_num_unit, 0, NULL, NULL );
	cl_ldu = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_ldu, CL_TRUE, 0,
								sizeof(int), &ldu, 0, NULL, NULL );
	cl_prev_ldu = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_prev_ldu, CL_TRUE, 0,
								sizeof(int), &prev_ldu, 0, NULL, NULL );

	cl_stride = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_stride, CL_TRUE, 0,
								sizeof(int), &stride, 0, NULL, NULL );
	cl_pad = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_pad, CL_TRUE, 0,
								sizeof(int), &pad, 0, NULL, NULL );
	cl_m = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_m, CL_TRUE, 0,
								sizeof(int), &m, 0, NULL, NULL );
	cl_n = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_n, CL_TRUE, 0,
								sizeof(int), &n, 0, NULL, NULL );
#endif
}

template<template<typename> class Mat, typename Real>
MaxPooling<Mat, Real>::~MaxPooling ()
{
#ifdef USE_GPU
	clReleaseMemObject( cl_num_unit );
	clReleaseMemObject( cl_prev_num_unit );
	clReleaseMemObject( cl_ldu );
	clReleaseMemObject( cl_prev_ldu );

	clReleaseMemObject( cl_stride );
	clReleaseMemObject( cl_pad );
	clReleaseMemObject( cl_m );
	clReleaseMemObject( cl_n );
#endif
}

template<template<typename> class Mat, typename Real>
#ifdef USE_MPI
void MaxPooling<Mat, Real>::init ( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world )
#else
void MaxPooling<Mat, Real>::init ( std::mt19937& m )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;

	MPI_Comm_rank(this->inner_world, &this->rank);
	MPI_Comm_size(this->inner_world, &this->nprocs);
#endif
	
	this->W = std::vector<Mat<Real>>();
	this->b = std::vector<Mat<Real>>();
}

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::finalize ()
{
	
}

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::calc_gradient ( const Mat<Real>& U_apply, const Mat<Real>& U_diff, const Mat<Real>& delta, std::vector<Mat<Real>>& nabla_W, std::vector<Mat<Real>>& nabla_b )
{
}

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::calc_delta ( const Matrix<Real>& U_apply, const Matrix<Real>& U_diff, const Matrix<Real>& delta, Matrix<Real>& nx_delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	my_size = (this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs;
	my_offset = this->rank*this->num_unit/this->nprocs;
#endif

	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	const int gap = prev_ldu + 2*pad;
	const int Y = this->prev_num_unit/prev_ldu, X = prev_ldu;

#pragma omp parallel for
	for( int i = 0; i < nx_delta.m*nx_delta.n; ++i ) nx_delta.v[i] = 0.0;
	for( int i = 0; i < this->prev_num_map; ++i ){
		beg = std::chrono::system_clock::now();

#pragma omp parallel for
		for( int j = 0; j < my_size; ++j ){
			int x = (j + my_offset)%ldu, y = (j + my_offset)/ldu;

			for( int k = 0; k < U_apply.n; ++k ){
				int s_idx = -1;
				Real val;
				
				for( int s = 0; s < m; ++s )
					for( int t = 0; t < n; ++t ){
						int idx = stride*x + t + s*gap + stride*y*gap;
						int nx = idx%gap - pad, ny = idx/gap - pad;

						if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;

						if( s_idx == -1 || val < U_apply(i*this->prev_num_unit + ny*prev_ldu + nx, k) ){
							val = U_apply(i*this->prev_num_unit + ny*prev_ldu + nx, k);
							s_idx = ny*prev_ldu + nx;
						}
					}
				
				nx_delta(i*this->prev_num_unit + s_idx, k) = delta(i*this->num_unit + j + my_offset, k) * U_diff(i*this->prev_num_unit + s_idx, k);
			}
		}
		end = std::chrono::system_clock::now();
		this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
	end = std::chrono::system_clock::now();
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &nx_delta(0,0), U_apply.m*U_apply.n, get_typecount(nx_delta(0,0)).mpi_type, MPI_SUM, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::calc_delta ( const clMatrix<Real>& U_apply, const clMatrix<Real>& U_diff, const clMatrix<Real>& delta, clMatrix<Real>& nx_delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	my_size = (this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs;
	my_offset = this->rank*this->num_unit/this->nprocs;
#endif

	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	cl_device_manager.set_argument( PRG::CLMAT_ZEROS, 0, &nx_delta.v );
	cl_device_manager.run_kernel( PRG::CLMAT_ZEROS, nx_delta.m*nx_delta.n, 1 );

	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 0, &nx_delta.v );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 1, &cl_prev_num_unit );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 2, &nx_delta.N );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 3, &U_apply.v );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 4, &U_diff.v );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 5, &delta.v );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 6, &cl_num_unit );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 7, &delta.N );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 8, &cl_prev_ldu );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 9, &cl_ldu );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 10, &cl_stride );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 11, &cl_pad );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 12, &cl_m );
	cl_device_manager.set_argument( PRG::MAXPOOL_DELTA, 13, &cl_n );
	cl_device_manager.run_kernel( PRG::MAXPOOL_DELTA, my_size, U_apply.n, this->prev_num_map );

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &nx_delta(0,0), U_apply.m*U_apply.n, get_typecount(nx_delta(0,0)).mpi_type, MPI_SUM, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	end = std::chrono::system_clock::now();
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}
#endif

template<template<typename> class Mat, typename Real>
Matrix<Real> MaxPooling<Mat, Real>::apply ( const Matrix<Real>& U, bool use_func )
{
	Matrix<Real> ret(this->num_map*this->num_unit, U.n);

	apply( U, ret, use_func );

	return ret;
}

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::apply ( const Matrix<Real>& U, Matrix<Real>& ret, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*my_size/this->nprocs - i*my_size/this->nprocs)*U.n;
		offset[i] = i*my_size/this->nprocs*U.n;
	}

	my_size = size[this->rank]/U.n;
	my_offset = offset[this->rank]/U.n;
#endif
	
	S = Matrix<Real>(this->num_map*this->num_unit, U.n);
	
	const int gap = prev_ldu + 2*pad;
	const int Y = this->prev_num_unit/prev_ldu, X = prev_ldu;

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	for( int i = 0; i < this->num_map; ++i ){
		beg = std::chrono::system_clock::now();

		Matrix<Real> tmp = Matrix<Real>::zeros(my_size, U.n);
#pragma omp parallel for
		for( int j = 0; j < my_size; ++j ){
			int x = (j + my_offset)%ldu, y = (j + my_offset)/ldu;

			for( int k = 0; k < U.n; ++k ){
				int s_idx = -1;
				Real val;

				for( int s = 0; s < m; ++s )
					for( int t = 0; t < n; ++t ){
						int idx = stride*x + t + s*gap + stride*y*gap;
						int nx = idx%gap - pad, ny = idx/gap - pad;
						
						if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;

						if( s_idx == -1 || val < U(i*this->prev_num_unit + ny*prev_ldu + nx, k) ){
							val = U(i*this->prev_num_unit + ny*prev_ldu + nx, k);
							s_idx = ny*prev_ldu + nx;
						}
					}

				tmp(j, k) = val;
				S(i*this->num_unit + j + my_offset, k) = s_idx;
			}
		}

		if( use_func ) this->func->inplace(tmp, false);
		end = std::chrono::system_clock::now();
		this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
#ifdef USE_MPI
		MPI_Allgatherv(&tmp(0,0), size[this->rank], get_typecount(tmp(0,0)).mpi_type,
					   &ret(i*this->num_unit,0), &size[0], &offset[0], get_typecount(ret(i*this->num_unit,0)).mpi_type, this->inner_world);
		MPI_Allgatherv(MPI_IN_PLACE, size[this->rank], get_typecount(S(0,0)).mpi_type,
					   &S(i*this->num_unit,0), &size[0], &offset[0], get_typecount(S(0,0)).mpi_type, this->inner_world);
#else
		ret.sub(i*this->num_unit, 0, tmp.m, tmp.n, tmp);
#endif
		end = std::chrono::system_clock::now();
		this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	end = std::chrono::system_clock::now();
	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> MaxPooling<Mat, Real>::apply ( const clMatrix<Real>& U, bool use_func )
{
	clMatrix<Real> ret(this->num_map*this->num_unit, U.n);

	apply( U, ret, use_func );

	return ret;
}

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::apply ( const clMatrix<Real>& U, clMatrix<Real>& ret, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*my_size/this->nprocs - i*my_size/this->nprocs)*U.n;
		offset[i] = i*my_size/this->nprocs*U.n;
	}

	my_size = size[this->rank]/U.n;
	my_offset = offset[this->rank]/U.n;
#endif
	
	S = clMatrix<Real>(this->num_map*this->num_unit, U.n);
	
	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 0, &ret.v );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 1, &cl_num_unit );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 2, &ret.N );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 3, &U.v );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 4, &S.v );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 5, &cl_prev_num_unit );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 6, &U.N );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 7, &cl_prev_ldu );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 8, &cl_ldu );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 9, &cl_stride );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 10, &cl_pad );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 11, &cl_m );
	cl_device_manager.set_argument( PRG::MAXPOOL_APPLY, 12, &cl_n );
	cl_device_manager.run_kernel( PRG::MAXPOOL_APPLY, my_size, U.n, this->prev_num_map );

	end = std::chrono::system_clock::now();
	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}
#endif

// std::vector<MaxPooling::Mat> MaxPooling::unpooling ( const std::vector<Mat>& U )
// {
// 	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
// 	std::vector<Mat> ret(num_map);

// 	int i,j,k,y,x;
// #pragma omp parallel for default(none) \
// 	private(i,j,y,x) shared(ret, U)
// 	for( i = 0; i < num_map; ++i ){
// 		ret[i] = Mat(prev_num_unit, U[0].n);
// 		for( j = 0; j < U[0].n; ++j ){
// 			for( y = 0; y < Y; y += stride )
// 				for( x = 0; x < X; x += stride ){
// 					int idx = S[i](x/stride + (y/stride)*ldu, j);
// 					double val = U[i](x+prev_ldu*y,j);

// 					ret[i](idx,j) = U[i](x/stride + (y/stride)*ldu,j);
// 				}
// 		}
// 	}

// 	return ret;
// }

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db )
{
	
}

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::set_W ( const std::string& filename )
{
	
}

template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::output_W ( const std::string& filename )
{

}

#ifdef USE_MPI
template<template<typename> class Mat, typename Real>
void MaxPooling<Mat, Real>::param_mix ()
{

}
#endif

#endif
