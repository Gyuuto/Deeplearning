#ifndef TRANSPOSEDCONVOLUTIONAL_HPP
#define TRANSPOSEDCONVOLUTIONAL_HPP

#include <fstream>
#include "Layer.hpp"

template<template<typename> class Mat, typename Real>
class TransposedConvolutional : public Layer<Mat, Real>
{
private:
	int prev_ldu, ldu;
	int m, n, stride, pad;
	int once_num;

	std::vector<int> feed_idx;
#ifdef USE_GPU
	cl_mem cl_feed_idx;
	cl_mem cl_num_unit, cl_prev_num_unit, cl_filter_size;
	cl_mem cl_my_size, cl_l_idx; // for future

	cl_mem cl_i, cl_k;
#endif
public:
	TransposedConvolutional( int prev_num_map, int prev_num_unit, int prev_ldu,
							 int num_map, int num_unit, int ldu,
							 int m, int n, int stride, int pad,
							 const std::shared_ptr<Function<Real>>& f, bool use_bias = true );
	~TransposedConvolutional ();

#ifdef USE_MPI
	void init( std::mt19937& mt, MPI_Comm inner_world, MPI_Comm outer_world );
#else
	void init( std::mt19937& mt );
#endif
	void finalize();
	
	std::pair<std::vector<Matrix<Real>>, std::vector<Matrix<Real>>> calc_gradient ( const Matrix<Real>& U, const Matrix<Real>& delta );
#ifdef USE_GPU
	std::pair<std::vector<clMatrix<Real>>, std::vector<clMatrix<Real>>> calc_gradient ( const clMatrix<Real>& U, const clMatrix<Real>& delta );
#endif

	Matrix<Real> calc_delta ( const Matrix<Real>& U, const Matrix<Real>& delta );
#ifdef USE_GPU
	clMatrix<Real> calc_delta ( const clMatrix<Real>& U, const clMatrix<Real>& delta );
#endif

	void update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db );

	Matrix<Real> apply ( const Matrix<Real>& U, bool use_func = true );
#ifdef USE_GPU
	clMatrix<Real> apply ( const clMatrix<Real>& U, bool use_func = true );
#endif

	// std::vector<Mat> deconvolution ( const std::vector<Mat>& U );

	void set_once_num ( const int& once_num );

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename );
	
#ifdef USE_MPI
	void param_mix ();
#endif
};

template<template<typename> class Mat, typename Real>
TransposedConvolutional<Mat, Real>::TransposedConvolutional( int prev_num_map, int prev_num_unit, int prev_ldu,
															 int num_map, int num_unit, int ldu,
															 int m, int n, int stride, int pad,
															 const std::shared_ptr<Function<Real>>& f, bool use_bias )
{
	this->once_num = 1;
	
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->prev_ldu = prev_ldu;
	
	this->num_map = num_map;
	this->num_unit = num_unit;
	this->ldu = ldu;	

	this->is_use_bias = use_bias;

	this->t_apply = this->t_delta = this->t_grad = 0.0;
	this->t_apply_init = this->t_apply_gemm = this->t_apply_repl = this->t_apply_comm = 0.0;
	this->t_delta_init = this->t_delta_gemm = this->t_delta_repl = this->t_delta_comm = 0.0;
	this->t_grad_init = this->t_grad_gemm = this->t_grad_repl = this->t_grad_comm = 0.0;

	this->m = m; this->n = n; this->stride = stride; this->pad = pad;

	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	if( num_unit%ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of output on Convolution layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( prev_num_unit%prev_ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of input on Convolution layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( ldu != (prev_ldu*stride + 2*pad - n) + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image width on Convolution layer.\n");
			printf("          Estimate width = %d.\n", (prev_ldu*stride + 2*pad - n) + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( num_unit/ldu != (prev_num_unit/prev_ldu*stride + 2*pad - m) + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image height on Convolution layer.\n");
			printf("          Estimate height = %d.\n", (prev_num_unit/prev_ldu*stride + 2*pad - m) + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	
	this->func = f;

	this->W.resize(1);
	this->W[0] = Mat<Real>(m*n*this->prev_num_map, this->num_map);

	this->b.resize(1);
	this->b[0] = Mat<Real>::zeros(this->num_map, 1);
}

template<template<typename> class Mat, typename Real>
TransposedConvolutional<Mat, Real>::~TransposedConvolutional()
{
#ifdef USE_GPU
	clReleaseMemObject( cl_feed_idx );

	clReleaseMemObject( cl_num_unit );
	clReleaseMemObject( cl_prev_num_unit );
	clReleaseMemObject( cl_filter_size );

	clReleaseMemObject( cl_l_idx );

	clReleaseMemObject( cl_i );
	clReleaseMemObject( cl_k );
#endif
}

template<template<typename> class Mat, typename Real>
#ifdef USE_MPI
void TransposedConvolutional<Mat, Real>::init ( std::mt19937& mt, MPI_Comm inner_world, MPI_Comm outer_world )
#else
	void TransposedConvolutional<Mat, Real>::init ( std::mt19937& mt )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;

	MPI_Comm_rank(inner_world, &this->rank);
	MPI_Comm_size(inner_world, &this->nprocs);
#endif

	// calculate indices of feed forward
	const int Y = this->prev_num_unit/prev_ldu*stride, X = prev_ldu*stride;
	const int gap = prev_ldu*stride + 2*pad;

	feed_idx.resize(this->num_unit*m*n);
#pragma omp parallel for
	for( int i = 0; i < this->num_unit; ++i ){
		int x = i%ldu, y = i/ldu;
		for( int s = 0; s < m; ++s )
			for( int t = 0; t < n; ++t ){
				int idx = x + t + s*gap + y*gap;
				int nx = idx%gap - pad, ny = idx/gap - pad;

				if( nx < 0 || nx >= X || ny < 0 || ny >= Y ){
					feed_idx[i*m*n + s*n + t] = -1;
					continue;
				}

				if( ny%stride != 0 || nx%stride != 0 ) feed_idx[i*m*n + s*n + t] = -1;
				else feed_idx[i*m*n + s*n + t] = ny/stride*prev_ldu + nx/stride;
			}
	}
	
#ifdef USE_GPU
	cl_int err;
	cl_feed_idx = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, this->num_unit*m*n*sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_feed_idx, CL_TRUE, 0,
								this->num_unit*m*n*sizeof(int), &feed_idx[0], 0, NULL, NULL );

	cl_num_unit = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_num_unit, CL_TRUE, 0,
								sizeof(int), &this->num_unit, 0, NULL, NULL );
	cl_prev_num_unit = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_prev_num_unit, CL_TRUE, 0,
								sizeof(int), &this->prev_num_unit, 0, NULL, NULL );
	cl_filter_size = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	int mn = m*n;
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_filter_size, CL_TRUE, 0,
								sizeof(int), &mn, 0, NULL, NULL );

	cl_l_idx = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	mn = 0;
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_l_idx, CL_TRUE, 0,
								sizeof(int), &mn, 0, NULL, NULL );

	cl_i = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	cl_k = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
#endif

	std::normal_distribution<Real> d_rand(0.0, 1.0E-1);
	Matrix<Real> tmp_W = this->W[0];
	for( int i = 0; i < tmp_W.m; ++i )
		for( int j = 0; j < tmp_W.n; ++j )
			tmp_W(i, j) = d_rand(mt);
	this->W[0] = tmp_W;
}

template<template<typename> class Mat, typename Real>
void TransposedConvolutional<Mat, Real>::finalize ()
{
}

template<template<typename> class Mat, typename Real>
std::pair<std::vector<Matrix<Real>>, std::vector<Matrix<Real>>> TransposedConvolutional<Mat, Real>::calc_gradient ( const Matrix<Real>& U, const Matrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0, my_size = this->num_unit;
#ifdef USE_MPI
	offset = this->rank*my_size/this->nprocs;
	my_size = (this->rank+1)*my_size/this->nprocs - this->rank*my_size/this->nprocs;
#endif

	std::vector<Matrix<Real>> nabla_W(1, Matrix<Real>::zeros(this->W[0].m, this->W[0].n));
	std::vector<Matrix<Real>> nabla_b(1, Matrix<Real>(this->num_map, 1));

	Matrix<Real> U_ = (*this->prev_func)(U, false);
	// const int prev_X = this->prev_ldu, prev_Y = this->prev_num_map/this->prev_ldu;
	// Matrix<Real> U_apply = (*this->prev_func)(U, false);
	// Matrix<Real> U_ = Matrix<Real>::zeros(this->prev_num_map*this->prev_num_unit*stride*stride, U.n);
	// for( int i = 0; i < this->prev_num_map; ++i ){
	// 	for( int y = 0; y < prev_Y; ++y )
	// 		for( int x = 0; x < prev_X; ++x )
	// 			for( int n = 0; n < U.n; ++n )
	// 				U_(i*this->prev_num_unit*stride*stride + y*prev_ldu*stride + x*stride, n) = U_apply(i*this->prev_num_unit + y*prev_ldu + x, n);
	// }
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	Matrix<Real> U_mat(m*n*this->prev_num_map, once_num*my_size), delta_mat(once_num*my_size, this->num_map);
	for( int i = 0; i < delta.n; i += once_num ){
		int size = std::min(once_num, delta.n - i);
		auto beg = std::chrono::system_clock::now();

#pragma omp parallel
		{
			for( int l = 0; l < size; ++l )
				for( int k = 0; k < this->prev_num_map; ++k )
#pragma omp for nowait
					for( int j = 0; j < my_size; ++j )
						for( int s = 0; s < m*n; ++s ){
							if( feed_idx[(j+offset)*m*n + s] != -1 )
								U_mat(k*m*n + s, j + l*my_size) = U_(k*this->prev_num_unit*stride*stride + feed_idx[(j+offset)*m*n + s], l+i);
							else
								U_mat(k*m*n + s, j + l*my_size) = 0.0;
						}

			for( int l = 0; l < size; ++l )
#pragma omp for nowait
				for( int k = 0; k < my_size; ++k )
					for( int j = 0; j < this->num_map; ++j ){
						delta_mat(l*my_size + k, j) = delta(j*this->num_unit + offset + k, l+i);
					}
		}
		auto end = std::chrono::system_clock::now();
		this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
		nabla_W[0] += U_mat*delta_mat;
		end = std::chrono::system_clock::now();
		this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	beg = std::chrono::system_clock::now();
	if( this->is_use_bias ){
		for( int i = 0; i < this->num_map; ++i ){
			double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
			for( int k = 0; k < my_size; ++k )
				for( int j = 0; j < delta.n; ++j )
					sum += delta(i*this->num_unit + offset + k, j);
				
			nabla_b[0](i,0) = sum;
		}
	}
	end = std::chrono::system_clock::now();
	this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	
	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &nabla_W[0](0,0), nabla_W[0].m*nabla_W[0].n, get_typecount(nabla_W[0](0,0)).mpi_type, MPI_SUM, this->inner_world);
	MPI_Allreduce(MPI_IN_PLACE, &nabla_b[0](0,0), nabla_b[0].m*nabla_b[0].n, get_typecount(nabla_b[0](0,0)).mpi_type, MPI_SUM, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_grad_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	end = std::chrono::system_clock::now();
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return std::make_pair(nabla_W, nabla_b);
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
std::pair<std::vector<clMatrix<Real>>, std::vector<clMatrix<Real>>> TransposedConvolutional<Mat, Real>::calc_gradient ( const clMatrix<Real>& U, const clMatrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0, my_size = this->num_unit;
#ifdef USE_MPI
	offset = this->rank*my_size/this->nprocs;
	my_size = (this->rank+1)*my_size/this->nprocs - this->rank*my_size/this->nprocs;
#endif

	std::vector<clMatrix<Real>> nabla_W(1, clMatrix<Real>::zeros(this->W[0].m, this->W[0].n));
	std::vector<clMatrix<Real>> nabla_b(1, clMatrix<Real>(this->num_map, 1));

	clMatrix<Real> U_ = (*this->prev_func)(U, false);
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

#ifdef USE_MPI
	const int tmp_size = (this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs; 
	const int tmp_offset = this->rank*this->num_unit/this->nprocs;
#else
	const int tmp_size = this->prev_num_unit;
	const int tmp_offset = 0;
#endif
	int l_idx = std::max(0, tmp_offset - m*prev_ldu/2);
	int r_idx = std::min(this->num_unit, tmp_offset + tmp_size + m*prev_ldu/2);

	cl_int err;

	clMatrix<Real> U_mat(m*n*this->prev_num_map, once_num*my_size), delta_mat(once_num*my_size, this->num_map);
	for( int i = 0; i < delta.n; i += once_num ){
		int size = std::min(once_num, delta.n - i);
		auto beg = std::chrono::system_clock::now();

		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_i, CL_TRUE, 0,
									sizeof(int), &i, 0, NULL, NULL );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 0, &U_mat.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 1, &U_mat.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 2, &U_.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 3, &cl_prev_num_unit );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 4, &U_.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 5, &cl_i );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 6, &cl_filter_size );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 7, &cl_l_idx );
		cl_device_manager.set_argument( PRG::CONV_GRAD_IMG_SET, 8, &cl_feed_idx );
		cl_device_manager.run_kernel( PRG::CONV_GRAD_IMG_SET, size, my_size, this->prev_num_map*m*n );

		cl_device_manager.set_argument( PRG::CONV_GRAD_DELTA_SET, 0, &delta_mat.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_DELTA_SET, 1, &delta_mat.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_DELTA_SET, 2, &delta.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_DELTA_SET, 3, &cl_num_unit );
		cl_device_manager.set_argument( PRG::CONV_GRAD_DELTA_SET, 4, &delta.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_DELTA_SET, 5, &cl_i );
		cl_device_manager.run_kernel( PRG::CONV_GRAD_DELTA_SET, this->num_map, size, my_size );
		auto end = std::chrono::system_clock::now();
		this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
		nabla_W[0] += U_mat*delta_mat;
		end = std::chrono::system_clock::now();
		this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	beg = std::chrono::system_clock::now();
	if( this->is_use_bias ){
		clMatrix<Real> buf(this->num_map, this->num_unit*delta.n);
		
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_HELPER, 0, &buf.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_HELPER, 1, &buf.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_HELPER, 2, &delta.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_HELPER, 3, &delta.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_HELPER, 4, &cl_num_unit );
		cl_device_manager.run_kernel( PRG::CONV_GRAD_BIAS_HELPER, delta.n, this->num_unit, this->num_map );
		
		const int minimum_once_num = 16;
		size_t lc1 = 1, lc2 = 1;

		for ( int i = 2; i*i < this->num_map; ++i ){
			if( i > cl_device_manager.get_max_work_group()/minimum_once_num ) break;
			if( this->num_map%i == 0 ){
				size_t x = i, y = this->num_map/i;
				if( y > cl_device_manager.get_max_work_group()/minimum_once_num ) y = 0;
			
				lc2 = std::max(lc2, (size_t)std::max(x, y));
			}
		}
		for ( int i = 2; i*i < delta.n*this->num_unit; ++i ){
			if( i > cl_device_manager.get_max_work_group()/lc2 ) break;
			if( delta.n*this->num_unit%i == 0 ){
				size_t x = i, y = delta.n*this->num_unit/i;
				if( y > cl_device_manager.get_max_work_group()/lc2 ) y = 0;
			
				lc1 = std::max(lc1, (size_t)std::max(x, y));
			}
		}

		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS, 0, &buf.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS, 1, &buf.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS, 2, cl_device_manager.get_max_work_group()*sizeof(Real) );
		cl_device_manager.run_kernel( PRG::CONV_GRAD_BIAS,
									  {(size_t)delta.n*this->num_unit, (size_t)this->num_map, 1},
									  {lc1, lc2, 1} );

		int n = (delta.n*this->num_unit)/lc1;
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_i, CL_TRUE, 0,
									sizeof(Real), &n, 0, NULL, NULL );
		
		
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_FINAL_REDUCE, 0, &buf.v );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_FINAL_REDUCE, 1, &buf.N );
		cl_device_manager.set_argument( PRG::CONV_GRAD_BIAS_FINAL_REDUCE, 2, &cl_i );
		cl_device_manager.run_kernel( PRG::CONV_GRAD_BIAS_FINAL_REDUCE, this->num_map );

		nabla_b[0].sub(0, 0, this->num_map, 1, buf);
	}
	end = std::chrono::system_clock::now();
	this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	
	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &nabla_W[0](0,0), nabla_W[0].m*nabla_W[0].n, get_typecount(nabla_W[0](0,0)).mpi_type, MPI_SUM, inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_grad_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	end = std::chrono::system_clock::now();
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return std::make_pair(nabla_W, nabla_b);
}
#endif

template<template<typename> class Mat, typename Real>
Matrix<Real> TransposedConvolutional<Mat, Real>::calc_delta ( const Matrix<Real>& U, const Matrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->prev_num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->prev_num_unit/this->nprocs - i*this->prev_num_unit/this->nprocs)*this->prev_num_map;
		offset[i] = i*this->prev_num_unit/this->nprocs*this->prev_num_map;
	}

	my_offset = offset[this->rank] / this->prev_num_map;
	my_size = size[this->rank] / this->prev_num_map;
#endif

	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	Matrix<Real> kernel(m*n*this->num_map, this->prev_num_map);
#pragma omp parallel
	{	
		for( int i = 0; i < this->num_map; ++i )
			for( int j = 0; j < this->prev_num_map; ++j )
#pragma omp for nowait
				for( int s = 0; s < m*n; ++s )
					kernel(i*m*n + s, j) = this->W[0](j*m*n + s, i);
	}
	
#ifdef USE_MPI
	const int tmp_size = (this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs; 
	const int tmp_offset = this->rank*this->num_unit/this->nprocs;
#else
	const int tmp_size = this->num_unit;
	const int tmp_offset = 0;
#endif
	int l_idx = std::max(0, tmp_offset - m*prev_ldu/2);
	int r_idx = std::min(this->num_unit, tmp_offset + tmp_size + m*prev_ldu/2);

	Matrix<Real> input_image(my_size*once_num, m*n*this->num_map);
	std::vector<Matrix<Real>> tmp_img((delta.n + once_num - 1)/once_num);
	for( int i = 0; i < delta.n; i += once_num ){
		int size = std::min(once_num, delta.n - i);
		auto beg = std::chrono::system_clock::now();

#pragma omp parallel for schedule(auto)
		for( int j = 0; j < my_size*once_num; ++j )
			for( int k = 0; k < m*n*this->num_map; ++k )
				input_image(j, k) = 0.0;

#pragma omp parallel
		{
			for( int l = 0; l < size; ++l )
#pragma omp for nowait
				for( int j = l_idx; j < r_idx; ++j )
					for( int k = 0; k < this->num_map; ++k )
						for( int s = 0; s < m*n; ++s ){
							if( feed_idx[j*m*n + s] != -1 &&
								my_offset <= feed_idx[j*m*n + s] && feed_idx[j*m*n + s] < my_offset + my_size )
								input_image(feed_idx[j*m*n + s] - my_offset + l*my_size, m*n*k + s) = delta(k*this->num_unit + j, l+i);
						}
			
		}
		auto end = std::chrono::system_clock::now();
		this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		
		beg = std::chrono::system_clock::now();
		tmp_img[i/once_num] = input_image * kernel;//this->W[0];
		end = std::chrono::system_clock::now();
		this->t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	Matrix<Real> nx_delta(this->prev_num_map*this->prev_num_unit, delta.n);
#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	Real* buf = new Real[delta.n*this->prev_num_unit*this->prev_num_map];

#pragma omp parallel
	{
		for( int i = 0; i < U.n; ++i )
#pragma omp for nowait
			for( int j = 0; j < this->prev_num_map; ++j )
				for( int k = 0; k < my_size; ++k )
					buf[i*(this->prev_num_map*my_size) + j*my_size + k + offset[this->rank]*U.n] = tmp_img[i/once_num](k + (i%once_num)*my_size, j);
	}

	std::vector<int> gath_size(this->nprocs), gath_displs(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		gath_size[i] = size[i]*U.n;
		gath_displs[i] = offset[i]*U.n;
	}

	MPI_Request req;
	MPI_Iallgatherv(MPI_IN_PLACE, gath_size[this->rank], get_typecount(buf[0]).mpi_type,
					buf, &gath_size[0], &gath_displs[0], get_typecount(buf[0]).mpi_type, this->inner_world, &req);
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int j = 0; j < this->prev_num_map; ++j )
#pragma omp for nowait
			for( int k = 0; k < size[this->rank]/this->prev_num_map; ++k )
				for( int i = 0; i < U.n; ++i )
					nx_delta(j*this->prev_num_unit + offset[this->rank]/this->prev_num_map+k, i) = buf[i*size[this->rank]+j*size[this->rank]/this->prev_num_map + k + offset[this->rank]*U.n];
	}

	MPI_Status stat;
	MPI_Wait(&req, &stat);
	
#pragma omp parallel
	{
		for( int j = 0; j < this->prev_num_map; ++j )
			for( int n = 0; n < this->nprocs; ++n ){
				if( n == this->rank ) continue;
#pragma omp for nowait
				for( int k = 0; k < size[n]/this->prev_num_map; ++k )
					for( int i = 0; i < U.n; ++i )
						nx_delta(j*this->prev_num_unit + offset[n]/this->prev_num_map+k, i) = buf[i*size[n]+j*size[n]/this->prev_num_map+k + offset[n]*U.n];
			}
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	delete [] buf;	
#else
	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int j = 0; j < this->prev_num_map; ++j )
#pragma omp for nowait
			for( int k = 0; k < my_size; ++k )
				for( int i = 0; i < U.n; ++i )
					nx_delta(j*this->prev_num_unit + k, i) = tmp_img[i/once_num](k + (i%once_num)*my_size, j);
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

	beg = std::chrono::system_clock::now();
	nx_delta = Matrix<Real>::hadamard(nx_delta, (*this->prev_func)(U, true));
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	end = std::chrono::system_clock::now();
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> TransposedConvolutional<Mat, Real>::calc_delta ( const clMatrix<Real>& U, const clMatrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->prev_num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->prev_num_unit/this->nprocs - i*this->prev_num_unit/this->nprocs)*this->prev_num_map;
		offset[i] = i*this->prev_num_unit/this->nprocs*this->prev_num_map;
	}

	my_offset = offset[this->rank] / this->prev_num_map;
	my_size = size[this->rank] / this->prev_num_map;
#endif

	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	const int gap = prev_ldu + 2*pad;
#ifdef USE_MPI
	const int tmp_size = (this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs; 
	const int tmp_offset = this->rank*this->num_unit/this->nprocs;
#else
	const int tmp_size = this->num_unit;
	const int tmp_offset = 0;
#endif
	int l_idx = std::max(0, tmp_offset - m*prev_ldu/2);
	int r_idx = std::min(this->num_unit, tmp_offset + tmp_size + m*prev_ldu/2);

	cl_int err;

	clMatrix<Real> kernel(m*n*this->num_map, this->prev_num_map);	
	cl_device_manager.set_argument( PRG::CONV_DELTA_KERNEL_SET, 0, &kernel.v );
	cl_device_manager.set_argument( PRG::CONV_DELTA_KERNEL_SET, 1, &this->W[0].v );
	cl_device_manager.run_kernel( PRG::CONV_DELTA_KERNEL_SET, this->prev_num_map, this->num_map, m*n );
	
	clMatrix<Real> input_image(my_size*once_num, m*n*this->num_map);
	std::vector<clMatrix<Real>> tmp_img(delta.n/once_num + 1);
	for( int i = 0; i < delta.n; i += once_num ){
		int size = std::min(once_num, delta.n - i);
		auto beg = std::chrono::system_clock::now();

		cl_device_manager.set_argument( PRG::CLMAT_ZEROS, 0, &input_image.v );
		cl_device_manager.run_kernel( PRG::CLMAT_ZEROS, input_image.m*input_image.n );
		
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_i, CL_TRUE, 0,
									sizeof(int), &i, 0, NULL, NULL );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 0, &input_image.v );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 1, &input_image.N );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 2, &delta.v );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 3, &cl_num_unit );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 4, &delta.N );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 5, &cl_i );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 6, &cl_filter_size );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 7, &cl_num_unit );
		cl_device_manager.set_argument( PRG::CONV_DELTA_IMG_SET, 8, &cl_feed_idx );
		cl_device_manager.run_kernel( PRG::CONV_DELTA_IMG_SET, this->num_map*m*n, size, my_size );
		auto end = std::chrono::system_clock::now();
		this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
		tmp_img[i/once_num] = input_image * kernel;
		end = std::chrono::system_clock::now();
		this->t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	clMatrix<Real> nx_delta(this->prev_num_map*this->prev_num_unit, delta.n);
#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	Real* buf = new Real[delta.n*this->prev_num_unit*this->prev_num_map];

#pragma omp parallel
	{
		for( int i = 0; i < U.n; ++i )
#pragma omp for nowait
			for( int j = 0; j < this->prev_num_map; ++j )
				for( int k = 0; k < my_size; ++k )
					buf[i*(this->prev_num_map*my_size) + j*my_size + k + offset[this->rank]*U[0].n] = tmp_img[i/once_num](k + (i%once_num)*my_size, j);
	}

	std::vector<int> gath_size(this->nprocs), gath_displs(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		gath_size[i] = size[i]*U.n;
		gath_displs[i] = offset[i]*U.n;
	}

	MPI_Request req;
	MPI_Iallgatherv(MPI_IN_PLACE, gath_size[this->rank], get_typecount(buf[0]).mpi_type,
					buf, &gath_size[0], &gath_displs[0], get_typecount(buf[0]).mpi_type, inner_world, &req);
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int j = 0; j < this->prev_num_map; ++j )
#pragma omp for nowait
			for( int k = 0; k < size[this->rank]/this->prev_num_map; ++k )
				for( int i = 0; i < U[0].n; ++i )
					nx_delta(j*this->prev_num_unit + offset[this->rank]/this->prev_num_map+k, i) = buf[i*size[this->rank]+j*size[this->rank]/this->prev_num_map + k + offset[this->rank]*U.n];
	}

	MPI_Status stat;
	MPI_Wait(&req, &stat);
	
#pragma omp parallel
	{
		for( int j = 0; j < this->prev_num_map; ++j )
			for( int n = 0; n < this->nprocs; ++n ){
				if( n == rank ) continue;
#pragma omp for nowait
				for( int k = 0; k < size[n]/this->prev_num_map; ++k )
					for( int i = 0; i < U.n; ++i )
						nx_delta(j*this->prev_num_unit + offset[n]/this->prev_num_map+k, i) = buf[i*size[n]+j*size[n]/this->prev_num_map+k + offset[n]*U.n];
			}
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	delete [] buf;	
#else
	beg = std::chrono::system_clock::now();

	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_i, CL_TRUE, 0,
								sizeof(int), &my_size, 0, NULL, NULL );
	for( int k = 0; k < tmp_img.size(); ++k ){
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_k, CL_TRUE, 0,
									sizeof(int), &k, 0, NULL, NULL );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 0, &nx_delta.v );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 1, &nx_delta.N );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 2, &tmp_img[k].v );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 3, &tmp_img[k].N );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 4, &cl_k );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 5, &cl_i );
		cl_device_manager.run_kernel( PRG::CONV_APPLY_RET_SET, once_num, this->prev_num_map, this->prev_num_unit );
	}	
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

	beg = std::chrono::system_clock::now();
	nx_delta = clMatrix<Real>::hadamard(nx_delta, (*this->prev_func)(U, true));
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	end = std::chrono::system_clock::now();
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}
#endif

template<template<typename> class Mat, typename Real>
void TransposedConvolutional<Mat, Real>::update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db)
{
	this->W[0] += dW[0];
	if( this->is_use_bias ) this->b[0] += db[0];
}

template<template<typename> class Mat, typename Real>
Matrix<Real> TransposedConvolutional<Mat, Real>::apply ( const Matrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){		
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*this->num_map;
		offset[i] = i*this->num_unit/this->nprocs*this->num_map;
	}

	my_offset = offset[this->rank] / this->num_map;
	my_size = size[this->rank] / this->num_map;
#endif
	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	Matrix<Real> input_image(my_size*once_num, m*n*this->prev_num_map);
	std::vector<Matrix<Real>> tmp_img(U.n/once_num + 1);
	for( int i = 0; i < U.n; i += once_num ){
		int size = std::min(once_num, U.n - i);

		auto beg = std::chrono::system_clock::now();
#pragma omp parallel
		{
			for( int l = 0; l < size; ++l )
#pragma omp for nowait
				for( int j = 0; j < my_size; ++j )
					for( int k = 0; k < this->prev_num_map; ++k )
						for( int s = 0; s < m*n; ++ s ){
							input_image(l*my_size + j, m*n*k + s) = (feed_idx[(j+my_offset)*m*n + s] != -1 ? U(k*this->prev_num_unit + feed_idx[(j+my_offset)*m*n + s], i+l) : 0.0);
						}
		}
		auto end = std::chrono::system_clock::now();
		this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
		tmp_img[i/once_num] = input_image * this->W[0];
		end = std::chrono::system_clock::now();
		this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	Matrix<Real> ret(this->num_map*this->num_unit, U.n);
#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	Real* buf = new Real[U.n*this->num_unit*this->num_map];

#pragma omp parallel
	{
		for( int i = 0; i < U.n; ++i )
#pragma omp for nowait
			for( int j = 0; j < this->num_map; ++j )
				for( int k = 0; k < my_size; ++k )
					buf[i*(this->num_map*my_size) + j*my_size + k + offset[this->rank]*U.n] = tmp_img[i/once_num](k + (i%once_num)*my_size, j);
	}

	std::vector<int> gath_size(this->nprocs), gath_displs(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		gath_size[i] = size[i]*U.n;
		gath_displs[i] = offset[i]*U.n;
	}

	MPI_Request req;
	MPI_Iallgatherv(MPI_IN_PLACE, gath_size[this->rank], get_typecount(buf[0]).mpi_type,
					buf, &gath_size[0], &gath_displs[0], get_typecount(buf[0]).mpi_type, this->inner_world, &req);
	end = std::chrono::system_clock::now();
	this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int j = 0; j < this->num_map; ++j )
#pragma omp for nowait
			for( int k = 0; k < size[this->rank]/this->num_map; ++k )
				for( int i = 0; i < U.n; ++i )
					ret(j*this->num_unit + offset[this->rank]/this->num_map+k, i) = buf[i*size[this->rank]+j*size[this->rank]/this->num_map+k + offset[this->rank]*U.n];
	}

	MPI_Status stat;
	MPI_Wait(&req, &stat);

#pragma omp parallel
	{
		for( int j = 0; j < this->num_map; ++j )
			for( int n = 0; n < this->nprocs; ++n ){
				if( n == this->rank ) continue;
#pragma omp for nowait
				for( int k = 0; k < size[n]/this->num_map; ++k )
					for( int i = 0; i < U.n; ++i )
						ret(j*this->num_unit + offset[n]/this->num_map+k, i) = buf[i*size[n]+j*size[n]/this->num_map+k + offset[n]*U.n];
			}
	}
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	delete [] buf;
#else
	beg = std::chrono::system_clock::now();

#pragma omp parallel
	{
		for( int j = 0; j < this->num_map; ++j )
#pragma omp for nowait
			for( int k = 0; k < this->num_unit; ++k )
				for( int i = 0; i < U.n; ++i )
					ret(j*this->num_unit + k, i) = tmp_img[i/once_num](k + (i%once_num)*my_size, j);
	}
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

	beg = std::chrono::system_clock::now();
	if( this->is_use_bias ){
#pragma omp parallel
		{
			for( int i = 0; i < this->num_map; ++i )
#pragma omp for nowait
				for( int j = 0; j < this->num_unit; ++j )
					for( int k = 0; k < ret.n; ++k )
						ret(i*this->num_unit + j, k) += this->b[0](i,0);
		}
	}

	if( use_func )
		ret = (*this->func)(ret, false);
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	end = std::chrono::system_clock::now();
	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return ret;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> TransposedConvolutional<Mat, Real>::apply ( const clMatrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){		
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*this->num_map;
		offset[i] = i*this->num_unit/this->nprocs*this->num_map;
	}

	my_offset = offset[rank] / this->num_map;
	my_size = size[rank] / this->num_map;
#endif
	cl_int err;

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	clMatrix<Real> input_image(my_size*once_num, m*n*this->prev_num_map);
	std::vector<clMatrix<Real>> tmp_img((U.n + once_num-1)/once_num);
	for( int i = 0; i < U.n; i += once_num ){
		int size = std::min(once_num, U.n - i);
		
		auto beg = std::chrono::system_clock::now();
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_i, CL_TRUE, 0,
									sizeof(int), &i, 0, NULL, NULL );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 0, &input_image.v );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 1, &input_image.N );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 2, &U.v );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 3, &cl_prev_num_unit );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 4, &U.N );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 5, &cl_i );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 6, &cl_filter_size );
		cl_device_manager.set_argument( PRG::CONV_APPLY_IMG_SET, 7, &cl_feed_idx );
		cl_device_manager.run_kernel( PRG::CONV_APPLY_IMG_SET, this->prev_num_map*m*n, size, my_size );
		auto end = std::chrono::system_clock::now();
		this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
		tmp_img[i/once_num] = input_image * this->W[0];
		end = std::chrono::system_clock::now();
		this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	clMatrix<Real> ret(this->num_map*this->num_unit, U.n);
#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	Real* buf = new Real[U.n*this->num_unit*this->num_map];

#pragma omp parallel
	{
		for( int i = 0; i < U.n; ++i )
#pragma omp for nowait
			for( int j = 0; j < this->num_map; ++j )
				for( int k = 0; k < my_size; ++k )
					buf[i*(this->num_map*my_size) + j*my_size + k + offset[doublerank]*U.n] = tmp_img[i/once_num](k + (i%once_num)*my_size, j);
	}

	std::vector<int> gath_size(doublenprocs), gath_displs(doublenprocs);
	for( int i = 0; i < doublenprocs; ++i ){
		gath_size[i] = size[i]*U.n;
		gath_displs[i] = offset[i]*U.n;
	}

	MPI_Request req;
	MPI_Iallgatherv(MPI_IN_PLACE, gath_size[doublerank], get_typecount(buf[0]).mpi_type,
					buf, &gath_size[0], &gath_displs[0], get_typecount(buf[0]).mpi_type, this->inner_world, &req);
	end = std::chrono::system_clock::now();
	this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int j = 0; j < this->num_map; ++j )
#pragma omp for nowait
			for( int k = 0; k < size[this->rank]/num_map; ++k )
				for( int i = 0; i < U.n; ++i )
					ret(j*this->num_unit + offset[this->rank]/this->num_map+k, i) = buf[i*size[this->rank]+j*size[this->rank]/this->num_map+k + offset[this->rank]*U.n];
	}

	MPI_Status stat;
	MPI_Wait(&req, &stat);

#pragma omp parallel
	{
		for( int j = 0; j < this->num_map; ++j )
			for( int n = 0; n < this->nprocs; ++n ){
				if( n == this->rank ) continue;
#pragma omp for nowait
				for( int k = 0; k < size[n]/this->num_map; ++k )
					for( int i = 0; i < U.n; ++i )
						ret(j*this->num_unit + offset[n]/this->num_map+k, i) = buf[i*size[n]+j*size[n]/this->num_map+k + offset[n]*U.n];
			}
	}
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	delete [] buf;
#else
	beg = std::chrono::system_clock::now();
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_i, CL_TRUE, 0,
								sizeof(int), &my_size, 0, NULL, NULL );
	for( int k = 0; k < tmp_img.size(); ++k ){
		err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_k, CL_TRUE, 0,
									sizeof(int), &k, 0, NULL, NULL );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 0, &ret.v );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 1, &ret.N );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 2, &tmp_img[k].v );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 3, &tmp_img[k].N );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 4, &cl_k );
		cl_device_manager.set_argument( PRG::CONV_APPLY_RET_SET, 5, &cl_i );
		cl_device_manager.run_kernel( PRG::CONV_APPLY_RET_SET, once_num, this->num_map, this->num_unit );
	}
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

	beg = std::chrono::system_clock::now();
	if( this->is_use_bias ){
		cl_device_manager.set_argument( PRG::CONV_APPLY_ADD_BIAS, 0, &ret.v );
		cl_device_manager.set_argument( PRG::CONV_APPLY_ADD_BIAS, 1, &this->b[0].v );
		cl_device_manager.run_kernel( PRG::CONV_APPLY_ADD_BIAS, this->num_unit*ret.n, this->num_map );
	}

	if( use_func )
		ret = (*this->func)(ret, false);
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	end = std::chrono::system_clock::now();
	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return ret;
}
#endif

// std::vector<TransposedConvolutional::Mat> TransposedConvolutional::deconvolution ( const std::vector<Mat>& U )
// {
// 	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
// 	std::vector<Mat> ret(prev_num_map);

// 	int i, j, k, x, y, s, t;
// #pragma omp parallel for default(none) \
// 	private(i,j,k,s,t,y,x) shared(ret, U)
// 	for( i = 0; i < prev_num_map; ++i ){
// 		ret[i] = Mat(prev_num_unit, U[0].n);
// 		for( j = 0; j < num_map; ++j ){
// 			auto U_ = (*func)(U[j], false);
// 			for( k = 0; k < U[0].n; ++k )
// 				for( x = 0; x < X; ++x )
// 					for( y = 0; y < Y; ++ y ){
// 						for( s = -m/2; s < (m+1)/2; ++s )
// 							for( t = -n/2; t < (n+1)/2; ++t ){
// 								int nx = (x - s),
// 									ny = (y - t);
// 								if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;
// 								nx /= stride; ny /= stride;
// 								ret[i](x+prev_ldu*y,k) += W[j][i](s+m/2,t+n/2)*(U_(nx+ldu*ny,k) - bias[j]);
// 							}
// 					}
// 		}
// 	}

// 	return ret;
// }

template<template<typename> class Mat, typename Real>
void TransposedConvolutional<Mat, Real>::set_once_num ( const int& once_num )
{
	this->once_num = once_num;
}

template<template<typename> class Mat, typename Real>
void TransposedConvolutional<Mat, Real>::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

	Matrix<Real> tmp_W = this->W[0];
	for( int i = 0; i < tmp_W.m; ++i )
		for( int j = 0; j < tmp_W.n; ++j )
			ifs.read((char*)&tmp_W(i,j), sizeof(tmp_W(i,j)));
	this->W[0] = tmp_W;

	Matrix<Real> tmp_b = this->b[0];
	for( int i = 0; i < this->num_map; ++i )
		ifs.read((char*)&tmp_b(0,i), sizeof(tmp_b(0,i)));
	this->b[0] = tmp_b;
}

template<template<typename> class Mat, typename Real>
void TransposedConvolutional<Mat, Real>::output_W ( const std::string& filename )
{
#ifdef USE_MPI
	if( this->rank == 0 ){
#endif
		std::ofstream ofs(filename, std::ios::binary);

		Matrix<Real> tmp_W = this->W[0];
		for( int i = 0; i < tmp_W.m; ++i )
			for( int j = 0; j < tmp_W.n; ++j )
				ofs.write((char*)&tmp_W(i,j), sizeof(tmp_W(i,j)));

		Matrix<Real> tmp_b = this->b[0];
		for( int i = 0; i < this->num_map; ++i )
			ofs.write((char*)&tmp_b(0,i), sizeof(tmp_b(0,i)));
#ifdef USE_MPI
	}
#endif
}

#ifdef USE_MPI
template<template<typename> class Mat, typename Real>
void TransposedConvolutional<Mat, Real>::param_mix ()
{
	int nprocs;
	MPI_Comm_size(this->outer_world, &nprocs);
	if( this->W.size() == 0 ) return;

	int cnt = this->W[0].m*this->W[0].n + this->b[0].m;
	std::vector<Real> w(cnt);

	Matrix<Real> tmp_W = this->W[0];
#pragma omp parallel for
	for( int i = 0; i < tmp_W.m; ++i )
		for( int j = 0; j < tmp_W.n; ++j ){
			int idx = i*tmp_W.n + j;
			w[idx] = tmp_W(i, j);
		}

	Matrix<Real> tmp_b = this->b[0];
#pragma omp parallel for
	for( int i = 0; i < tmp_b.m; ++i ){
		int idx = tmp_W.m*tmp_W.n + i;
		w[idx] = tmp_b(i,0);
	}

	MPI_Allreduce(MPI_IN_PLACE, &w[0], cnt, get_typecount(w[0]).mpi_type, MPI_SUM, this->outer_world);

#pragma omp parallel for
	for( int i = 0; i < tmp_W.m; ++i )
		for( int j = 0; j < tmp_W.n; ++j ){
			int idx = i*tmp_W.n + j;
			tmp_W(i,j) = w[idx] / nprocs;
		}
	this->W[0] = tmp_W;
	
#pragma omp parallel for
	for( int i = 0; i < tmp_b.m; ++i ){
		int idx = tmp_W.m*tmp_W.n + i;
		tmp_b(i,0) = w[idx]/nprocs;
	}
	this->b[0] = tmp_b;
}
#endif

#endif
