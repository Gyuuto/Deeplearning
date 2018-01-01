#ifndef BATCHNORMALIZE_HPP
#define BATCHNORMALIZE_HPP

#include <fstream>
#include "Layer.hpp"

template<template<typename> class Mat, typename Real>
class BatchNormalize : public Layer<Mat, Real>
{
private:
	Real EPS, decay;
#ifdef USE_GPU
	cl_mem cl_EPS;
	cl_mem cl_num_unit, cl_prev_num_unit;

	cl_mem cl_n;
#endif

	Mat<Real> tmp_mean, tmp_var;
	Mat<Real> mean, var;
public:
	BatchNormalize( int prev_num_map, int prev_num_unit,
					const std::shared_ptr<Function<Real>>& f, const Real decay = 0.999, const Real EPS = 1.0E-8 );
	~BatchNormalize ();

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

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename );
	
#ifdef USE_MPI
	void param_mix ();
#endif
};

template<template<typename> class Mat, typename Real>
BatchNormalize<Mat, Real>::BatchNormalize( int prev_num_map, int prev_num_unit,
										   const std::shared_ptr<Function<Real>>& f, const Real decay, const Real EPS ) :decay(decay), EPS(EPS)
{
	this->layer_name = "BatchNormalize";

	this->prev_num_map = this->num_map = prev_num_map;
	this->prev_num_unit = this->num_unit = prev_num_unit;
	this->func = f;

	this->t_apply = this->t_delta = this->t_grad = 0.0;
	this->t_apply_init = this->t_apply_gemm = this->t_apply_repl = this->t_apply_comm = 0.0;
	this->t_delta_init = this->t_delta_gemm = this->t_delta_repl = this->t_delta_comm = 0.0;
	this->t_grad_init = this->t_grad_gemm = this->t_grad_repl = this->t_grad_comm = 0.0;

	this->W = std::vector<Mat<Real>>(1, Mat<Real>(this->num_map, 1));
	this->b = std::vector<Mat<Real>>(1, Mat<Real>(this->num_map, 1));
}

template<template<typename> class Mat, typename Real>
BatchNormalize<Mat, Real>::~BatchNormalize()
{
#ifdef USE_GPU
	clReleaseMemObject( cl_EPS );

	clReleaseMemObject( cl_num_unit );
	clReleaseMemObject( cl_prev_num_unit );

	clReleaseMemObject(cl_n);
#endif
}

template<template<typename> class Mat, typename Real>
#ifdef USE_MPI
void BatchNormalize<Mat, Real>::init( std::mt19937& m, MPI_Comm outer_world, MPI_Comm inner_world )
#else
void BatchNormalize<Mat, Real>::init( std::mt19937& m )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;
	MPI_Comm_size(this->inner_world, &this->nprocs);
	MPI_Comm_rank(this->inner_world, &this->rank);
#endif

	int seed = time(NULL);
#ifdef USE_MPI
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, this->inner_world);
#endif
	
	int offset, my_size;
#ifdef USE_MPI
	my_size = (this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs;
	offset = this->rank*this->num_unit/this->nprocs;
#else
	offset = 0;
	my_size = this->num_unit;
#endif
	mean = Mat<Real>::zeros(this->num_map, my_size); var = Mat<Real>::ones(this->num_map, my_size);
	tmp_mean = Mat<Real>(this->num_map, my_size); tmp_var = Mat<Real>(this->num_map, my_size);

	std::uniform_real_distribution<Real> d_rand(-1.0, 1.0);
	Matrix<Real> tmp_W = this->W[0], tmp_b = this->b[0];
	for( int i = 0; i < this->num_map; ++i ){
		tmp_W(i,0) = d_rand(m);
		tmp_b(i,0) = d_rand(m);
	}
	this->W[0] = tmp_W;
	this->b[0] = tmp_b;

#ifdef USE_GPU
	cl_int err;

	cl_EPS = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_EPS, CL_TRUE, 0,
								sizeof(Real), &EPS, 0, NULL, NULL );
	
	cl_num_unit = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_num_unit, CL_TRUE, 0,
								sizeof(int), &this->num_unit, 0, NULL, NULL );
	cl_prev_num_unit = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_prev_num_unit, CL_TRUE, 0,
								sizeof(int), &this->prev_num_unit, 0, NULL, NULL );

	cl_n = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
#endif
}

template<template<typename> class Mat, typename Real>
void BatchNormalize<Mat, Real>::finalize ()
{

}

template<template<typename> class Mat, typename Real>
std::pair<std::vector<Matrix<Real>>, std::vector<Matrix<Real>>> BatchNormalize<Mat, Real>::calc_gradient ( const Matrix<Real>& U, const Matrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;
	int my_offset, my_size;
#ifdef USE_MPI
	my_size = (this->rank+1)*this->prev_num_unit/this->nprocs - this->rank*this->prev_num_unit/this->nprocs;
	my_offset = this->rank*this->prev_num_unit/this->nprocs;
#else
	my_offset = 0;
	my_size = this->prev_num_unit;
#endif
	std::vector<Matrix<Real>> nabla_W(1, Matrix<Real>(this->num_map, 1));
	std::vector<Matrix<Real>> nabla_b(1, Matrix<Real>(this->num_map, 1));
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	auto U_apply = (*this->prev_func)(U, false);
	for( int i = 0; i < this->num_map; ++i ){
		Real tmp_nabla1 = 0.0, tmp_nabla2 = 0.0;
		auto beg = std::chrono::system_clock::now();

#pragma omp parallel for reduction(+:tmp_nabla1)
		for( int j = 0; j < my_size; ++j ){
			for( int k = 0; k < U.n; ++k ){
				tmp_nabla1 += delta(i*this->num_unit + my_offset+j,k)*(U_apply(i*this->prev_num_unit + my_offset+j,k) - tmp_mean(i,j))/std::sqrt(tmp_var(i,j) + EPS);
			}
		}

#pragma omp parallel for reduction(+:tmp_nabla2)
		for( int j = 0; j < my_size; ++j ){
			for( int k = 0; k < U.n; ++k ){
				tmp_nabla2 += delta(i*this->num_unit + my_offset+j,k);
			}
		}
		auto end = std::chrono::system_clock::now();
		this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
#ifdef USE_MPI
		MPI_Allreduce(MPI_IN_PLACE, &tmp_nabla1, 1, get_typecount(tmp_nabla1).mpi_type, MPI_SUM, this->inner_world);
		MPI_Allreduce(MPI_IN_PLACE, &tmp_nabla2, 1, get_typecount(tmp_nabla2).mpi_type, MPI_SUM, this->inner_world);
#endif
		end = std::chrono::system_clock::now();
		this->t_grad_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		
		nabla_W[0](i,0) = tmp_nabla1; nabla_b[0](i,0) = tmp_nabla2;
	}
	end = std::chrono::system_clock::now();
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	cnt_flop += this->num_map*(my_size*U.n*5 + my_size*U.n*1);
	
	return std::make_pair(nabla_W, nabla_b);
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
std::pair<std::vector<clMatrix<Real>>, std::vector<clMatrix<Real>>> BatchNormalize<Mat, Real>::calc_gradient ( const clMatrix<Real>& U, const clMatrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;
	int my_offset, my_size;
#ifdef USE_MPI
	my_size = (this->rank+1)*this->prev_num_unit/this->nprocs - this->rank*this->prev_num_unit/this->nprocs;
	my_offset = this->rank*this->prev_num_unit/this->nprocs;
#else
	my_offset = 0;
	my_size = this->prev_num_unit;
#endif
	std::vector<clMatrix<Real>> nabla_W(1, clMatrix<Real>(this->num_map, 1));
	std::vector<clMatrix<Real>> nabla_b(1, clMatrix<Real>(this->num_map, 1));
	clMatrix<Real> tmp_nabla1 = clMatrix<Real>::zeros(this->num_map, U.n*my_size);
	clMatrix<Real> tmp_nabla2 = clMatrix<Real>::zeros(this->num_map, U.n*my_size);
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	auto U_apply = (*this->prev_func)(U, false);

	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 0, &tmp_nabla1.v );
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 1, &tmp_nabla1.N );
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 2, &tmp_nabla2.v );
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 3, &tmp_nabla2.N );
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 4, &delta.v );
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 5, &U_apply.v );
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 6, &cl_num_unit );
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 7, &U.N );
	if( this->is_learning ){
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 8, &tmp_mean.v );
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 9, &tmp_mean.N );
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 10, &tmp_var.v );
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 11, &tmp_var.N );
	}
	else{
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 8, &mean.v );
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 9, &mean.N );
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 10, &var.v );
		cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 11, &var.N );
	}
	cl_device_manager.set_argument( PRG::BN_GRAD_HELPER, 12, &cl_EPS );
	cl_device_manager.run_kernel( PRG::BN_GRAD_HELPER, U.n, my_size, this->num_map );

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
	for ( int i = 2; i*i < U.n*my_size; ++i ){
		if( i > cl_device_manager.get_max_work_group()/lc2 ) break;
		if( U.n*my_size%i == 0 ){
			size_t x = i, y = U.n*my_size/i;
			if( y > cl_device_manager.get_max_work_group()/lc2 ) y = 0;
			
			lc1 = std::max(lc1, (size_t)std::max(x, y));
		}
	}
	
	cl_device_manager.set_argument( PRG::BN_GRAD, 0, &tmp_nabla1.v );
	cl_device_manager.set_argument( PRG::BN_GRAD, 1, &tmp_nabla1.N );
	cl_device_manager.set_argument( PRG::BN_GRAD, 2, &tmp_nabla2.v );
	cl_device_manager.set_argument( PRG::BN_GRAD, 3, &tmp_nabla2.N );
	cl_device_manager.set_argument( PRG::BN_GRAD, 4, lc1*lc2*sizeof(Real) );
	cl_device_manager.run_kernel( PRG::BN_GRAD,
								  {(size_t)U.n*my_size, (size_t)this->num_map, 1},
								  {lc1, lc2, 1} );

	cl_int err;
	int n = U.n*my_size / lc1;
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_n, CL_TRUE, 0,
								sizeof(int), &n, 0, NULL, NULL );

	cl_device_manager.set_argument( PRG::BN_GRAD_FINAL_REDUCE, 0, &tmp_nabla1.v );
	cl_device_manager.set_argument( PRG::BN_GRAD_FINAL_REDUCE, 1, &tmp_nabla1.N );
	cl_device_manager.set_argument( PRG::BN_GRAD_FINAL_REDUCE, 2, &tmp_nabla2.v );
	cl_device_manager.set_argument( PRG::BN_GRAD_FINAL_REDUCE, 3, &tmp_nabla2.N );
	cl_device_manager.set_argument( PRG::BN_GRAD_FINAL_REDUCE, 4, &cl_n );
	cl_device_manager.run_kernel( PRG::BN_GRAD_FINAL_REDUCE, this->num_map );

	nabla_W[0].sub(0, 0, this->num_map, 1, tmp_nabla1);
	nabla_b[0].sub(0, 0, this->num_map, 1, tmp_nabla2);
	
	// cl_int err;
	// cl_event event;
	// cl_mem scratch_buf = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_WRITE, U.n*my_size*sizeof(Real), NULL, &err );
	// for( int i = 0; i < this->num_map; ++i ){
	// 	clblasSdot( U.n*my_size, nabla_W[0].v, i, tmp_nabla1.v, i*U.n*my_size, 1, E.v, 0, 1, scratch_buf, 1,
	// 				cl_device_manager.get_queue_ptr(), 0, NULL, &event );
	// 	clReleaseEvent(event);

	// 	clblasSdot( U.n*my_size, nabla_b[0].v, i, tmp_nabla2.v, i*U.n*my_size, 1, E.v, 0, 1, scratch_buf, 1,
	// 				cl_device_manager.get_queue_ptr(), 0, NULL, &event );
	// 	clReleaseEvent(event);
	// }

	// clReleaseMemObject( scratch_buf );

#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &nabla_W[0](0,0), this->num_map, get_typecount(nabla_W[0](0,0)).mpi_type, MPI_SUM, this->inner_world);
	MPI_Allreduce(MPI_IN_PLACE, &nabla_b[0](0,0), this->num_map, get_typecount(nabla_b[0](0,0)).mpi_type, MPI_SUM, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	cnt_flop += this->num_map*(my_size*U.n*5 + my_size*U.n*1);
	
	return std::make_pair(nabla_W, nabla_b);
}
#endif

template<template<typename> class Mat, typename Real>
Matrix<Real> BatchNormalize<Mat, Real>::calc_delta ( const Matrix<Real>& U, const Matrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_offset, my_size;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->prev_num_unit/this->nprocs - i*this->prev_num_unit/this->nprocs)*U.n;
		offset[i] = i*this->prev_num_unit/this->nprocs*U.n;
	}
	my_size = size[this->rank]/U.n;
	my_offset = offset[this->rank]/U.n;
#else
	my_offset = 0;
	my_size = this->prev_num_unit;
#endif

	Matrix<Real> nx_delta(this->prev_num_map*this->prev_num_unit, delta.n);
#ifdef USE_MPI
	Matrix<Real> tmp_nx_delta(this->prev_num_map*my_size, delta.n);
#endif
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	auto U_appl = (*this->prev_func)(U, false);
	auto U_diff = (*this->prev_func)(U, true);
	for( int i = 0; i < this->num_map; ++i ){
		auto beg = std::chrono::system_clock::now();
		
#pragma omp parallel for 
		for( int j = 0; j < my_size; ++j ){
			Real tmp1 = 0.0, tmp2 = 0.0;
			for( int l = 0; l < U.n; ++l ){
				tmp1 += delta(i*this->num_unit + my_offset+j,l);
				tmp2 += delta(i*this->num_unit + my_offset+j,l)*(U_appl(i*this->prev_num_unit + my_offset+j,l) - tmp_mean(i,j));
			}
			tmp1 /= U.n; tmp2 /= U.n;
			
			for( int k = 0; k < U.n; ++k ){

#ifdef USE_MPI
				tmp_nx_delta(i*my_size + j,k) =
					this->W[0](i,0)/sqrt(tmp_var(i,j) + EPS)*delta(i*this->num_unit + my_offset+j,k)*U_diff(i*this->prev_num_unit + my_offset+j,k)
					- this->W[0](i,0)/sqrt(tmp_var(i,j) + EPS)*U_diff(i*this->prev_num_unit + my_offset+j,k)*tmp1
					- this->W[0](i,0)/(pow(tmp_var(i,j), 1.5) + EPS)*U_diff(i*this->prev_num_unit + my_offset+j,k)*(U_appl(i*this->prev_num_unit + my_offset+j,k) - tmp_mean(i,j))*tmp2;
#else
				nx_delta(i*this->prev_num_unit + j,k) =
					this->W[0](i,0)/sqrt(tmp_var(i,j) + EPS)*delta(i*this->num_unit + j,k)*U_diff(i*this->prev_num_unit + j,k)
					- this->W[0](i,0)/sqrt(tmp_var(i,j) + EPS)*U_diff(i*this->prev_num_unit + j,k)*tmp1
					- this->W[0](i,0)/(pow(tmp_var(i,j), 1.5) + EPS)*U_diff(i*this->prev_num_unit + j,k)*(U_appl(i*this->prev_num_unit + j,k) - tmp_mean(i,j))*tmp2;
#endif
			}
			auto end = std::chrono::system_clock::now();
			this->t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		}
	}

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	for( int i = 0; i < this->num_map; ++i )
		MPI_Allgatherv(&tmp_nx_delta(i*my_size,0), size[this->rank], get_typecount(tmp_nx_delta(i*my_size,0)).mpi_type,
					   &nx_delta(i*this->prev_num_unit,0), &size[0], &offset[0], get_typecount(nx_delta(i*this->prev_num_unit,0)).mpi_type, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	cnt_flop += this->num_map*(my_size*U.n*(U.n*4 + 2 + 19));

	return nx_delta;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> BatchNormalize<Mat, Real>::calc_delta ( const clMatrix<Real>& U, const clMatrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_offset, my_size;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->prev_num_unit/this->nprocs - i*this->prev_num_unit/this->nprocs)*U.n;
		offset[i] = i*this->prev_num_unit/this->nprocs*U.n;
	}
	my_size = size[this->rank]/U.n;
	my_offset = offset[this->rank]/U.n;
#else
	my_offset = 0;
	my_size = this->prev_num_unit;
#endif

	clMatrix<Real> nx_delta(this->prev_num_map*this->prev_num_unit, delta.n);
#ifdef USE_MPI
	clMatrix<Real> tmp_nx_delta(this->prev_num_map*my_size, delta.n);
#endif
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	auto U_apply = (*this->prev_func)(U, false);
	auto U_diff = (*this->prev_func)(U, true);

	cl_device_manager.set_argument( PRG::BN_DELTA, 0, &nx_delta.v );
	cl_device_manager.set_argument( PRG::BN_DELTA, 1, &cl_prev_num_unit );
	cl_device_manager.set_argument( PRG::BN_DELTA, 2, &nx_delta.N );
	cl_device_manager.set_argument( PRG::BN_DELTA, 3, &U_apply.v );
	cl_device_manager.set_argument( PRG::BN_DELTA, 4, &U_diff.v );
	cl_device_manager.set_argument( PRG::BN_DELTA, 5, &delta.v );
	cl_device_manager.set_argument( PRG::BN_DELTA, 6, &U.N );
	if( this->is_learning ){
		cl_device_manager.set_argument( PRG::BN_DELTA, 7, &tmp_mean.v );
		cl_device_manager.set_argument( PRG::BN_DELTA, 8, &tmp_mean.N );
		cl_device_manager.set_argument( PRG::BN_DELTA, 9, &tmp_var.v );
		cl_device_manager.set_argument( PRG::BN_DELTA, 10, &tmp_var.N );
	}
	else{
		cl_device_manager.set_argument( PRG::BN_DELTA, 7, &mean.v );
		cl_device_manager.set_argument( PRG::BN_DELTA, 8, &mean.N );
		cl_device_manager.set_argument( PRG::BN_DELTA, 9, &var.v );
		cl_device_manager.set_argument( PRG::BN_DELTA, 10, &var.N );
	}
	cl_device_manager.set_argument( PRG::BN_DELTA, 11, &this->W[0].v );
	cl_device_manager.set_argument( PRG::BN_DELTA, 12, &cl_EPS );
	cl_device_manager.run_kernel( PRG::BN_DELTA, U.n, my_size, this->num_map );

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	for( int i = 0; i < this->num_map; ++i )
		MPI_Allgatherv(&tmp_nx_delta(i*my_size,0), size[this->rank], get_typecount(tmp_nx_delta(i*my_size,0)).mpi_type,
					   &nx_delta(i*this->prev_num_unit,0), &size[0], &offset[0], get_typecount(nx_delta(i*this->prev_num_unit,0)).mpi_type, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	cnt_flop += this->num_map*(my_size*U.n*(U.n*4 + 2 + 19));

	return nx_delta;
}
#endif

template<template<typename> class Mat, typename Real>
void BatchNormalize<Mat, Real>::update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db )
{
	this->W[0] += dW[0]; this->b[0] += db[0];
}

template<template<typename> class Mat, typename Real>
Matrix<Real> BatchNormalize<Mat, Real>::apply ( const Matrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_offset, my_size;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_unit/this->nprocs*U.n;
	}
	my_size = size[this->rank]/U.n;
	my_offset = offset[this->rank]/U.n;
#else
	my_offset = 0;
	my_size = this->num_unit;
#endif
	tmp_mean = tmp_var = Matrix<Real>::zeros(this->num_map, my_size);

	Matrix<Real> ret(this->num_map*this->num_unit, U.n);
#ifdef USE_MPI
	Matrix<Real> tmp_ret(this->num_map*my_size, U.n);
#endif
	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();

	if( this->is_learning ){
#pragma omp parallel
		{
			for( int i = 0; i < this->num_map; ++i ){
#pragma omp for nowait
				for( int j = 0; j < my_size; ++j ){
					for( int k = 0; k < U.n; ++k ){
						tmp_mean(i,j) += U(i*this->prev_num_unit + my_offset+j, k);
					}
				}
			}
		}
		tmp_mean /= U.n;
	
#pragma omp parallel
		{
			for( int i = 0; i < this->num_map; ++i ){
#pragma omp for nowait
				for( int j = 0; j < my_size; ++j ){
					for( int k = 0; k < U.n; ++k ){
						Real v = (U(i*this->prev_num_unit + my_offset+j, k) - tmp_mean(i, j));
						tmp_var(i,j) += v*v;
					}
				}
			}
		}
		tmp_var /= U.n;

		mean -= (1 - decay) * (mean - tmp_mean);
		var -= (1 - decay) * (var - tmp_var);
	}
	
	if( this->is_learning ){
#pragma omp parallel
		{
			for( int i = 0; i < this->num_map; ++i ){
#pragma omp for nowait
				for( int j = 0; j < my_size; ++j ){
					for( int k = 0; k < U.n; ++k ){
#ifdef USE_MPI
						tmp_ret(i*my_size + j, k) = this->W[0](i,0)*(U(i*this->prev_num_unit + my_offset+j,k) - tmp_mean(i,j))/std::sqrt(tmp_var(i,j)+EPS) + this->b[0](i,0);
#else
						ret(i*this->num_unit + j, k) = this->W[0](i,0)*(U(i*this->prev_num_unit + j,k) - tmp_mean(i,j))/std::sqrt(tmp_var(i,j)+EPS) + this->b[0](i,0);
#endif					
					}
				}
			}
		}
	}
	else{
#pragma omp parallel
		{
			for( int i = 0; i < this->num_map; ++i ){
#pragma omp for nowait
				for( int j = 0; j < my_size; ++j ){
					for( int k = 0; k < U.n; ++k ){
#ifdef USE_MPI
						tmp_ret(i*my_size + j, k) = this->W[0](i,0)*(U(i*this->prev_num_unit + my_offset+j,k) - mean(i,j))/std::sqrt(var(i,j)+EPS) + this->b[0](i,0);
#else
						ret(i*this->num_unit + j, k) = this->W[0](i,0)*(U(i*this->prev_num_unit + j,k) - mean(i,j))/std::sqrt(var(i,j)+EPS) + this->b[0](i,0);
#endif					
					}
				}
			}
		}
	}		
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	for( int i = 0; i < this->num_map; ++i ){
		MPI_Allgatherv(&tmp_ret(i*my_size,0), size[this->rank], get_typecount(tmp_ret(i*my_size,0)).mpi_type,
					   &ret(i*this->num_unit,0), &size[0], &offset[0], get_typecount(ret(i*this->num_unit,0)).mpi_type, this->inner_world);
	}
#endif
	end = std::chrono::system_clock::now();
	this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	if( use_func ){
		beg = std::chrono::system_clock::now();
		ret = (*this->func)(ret, false);
		end = std::chrono::system_clock::now();
		this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	cnt_flop += this->num_map*U.n*my_size + this->num_map*U.n*my_size*3 + this->num_map*U.n*my_size*5;

	return ret;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> BatchNormalize<Mat, Real>::apply ( const clMatrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_offset, my_size;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_unit/this->nprocs*U.n;
	}
	my_size = size[this->rank]/U.n;
	my_offset = offset[this->rank]/U.n;
#else
	my_offset = 0;
	my_size = this->num_unit;
#endif

	clMatrix<Real> ret(this->num_map*this->num_unit, U.n);
#ifdef USE_MPI
	clMatrix<Real> tmp_ret(this->num_map*my_size, U.n);
#endif
	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();

	if( this->is_learning ){
		cl_device_manager.set_argument( PRG::BN_APPLY_MEAN_VAR, 0, &tmp_mean.v );
		cl_device_manager.set_argument( PRG::BN_APPLY_MEAN_VAR, 1, &tmp_mean.N );
		cl_device_manager.set_argument( PRG::BN_APPLY_MEAN_VAR, 2, &tmp_var.v );
		cl_device_manager.set_argument( PRG::BN_APPLY_MEAN_VAR, 3, &tmp_var.N );
		cl_device_manager.set_argument( PRG::BN_APPLY_MEAN_VAR, 4, &U.v );
		cl_device_manager.set_argument( PRG::BN_APPLY_MEAN_VAR, 5, &cl_prev_num_unit );
		cl_device_manager.set_argument( PRG::BN_APPLY_MEAN_VAR, 6, &U.N );
		cl_device_manager.run_kernel( PRG::BN_APPLY_MEAN_VAR, my_size, this->num_map );

		mean -= (1 - decay) * (mean - tmp_mean);
		var -= (1 - decay) * (var - tmp_var);
	}

	if( this->is_learning ){
		cl_device_manager.set_argument( PRG::BN_APPLY, 0, &ret.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 1, &cl_num_unit );
		cl_device_manager.set_argument( PRG::BN_APPLY, 2, &ret.N );
		cl_device_manager.set_argument( PRG::BN_APPLY, 3, &tmp_mean.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 4, &tmp_mean.N );
		cl_device_manager.set_argument( PRG::BN_APPLY, 5, &tmp_var.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 6, &tmp_var.N );
		cl_device_manager.set_argument( PRG::BN_APPLY, 7, &this->W[0].v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 8, &this->b[0].v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 9, &cl_EPS );
		cl_device_manager.set_argument( PRG::BN_APPLY, 10, &U.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 11, &cl_prev_num_unit );
		cl_device_manager.set_argument( PRG::BN_APPLY, 12, &U.N );
		cl_device_manager.run_kernel( PRG::BN_APPLY, U.n, my_size, this->num_map );
	}
	else{
		cl_device_manager.set_argument( PRG::BN_APPLY, 0, &ret.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 1, &cl_num_unit );
		cl_device_manager.set_argument( PRG::BN_APPLY, 2, &ret.N );
		cl_device_manager.set_argument( PRG::BN_APPLY, 3, &mean.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 4, &mean.N );
		cl_device_manager.set_argument( PRG::BN_APPLY, 5, &var.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 6, &var.N );
		cl_device_manager.set_argument( PRG::BN_APPLY, 7, &this->W[0].v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 8, &this->b[0].v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 9, &cl_EPS );
		cl_device_manager.set_argument( PRG::BN_APPLY, 10, &U.v );
		cl_device_manager.set_argument( PRG::BN_APPLY, 11, &cl_prev_num_unit );
		cl_device_manager.set_argument( PRG::BN_APPLY, 12, &U.N );
		cl_device_manager.run_kernel( PRG::BN_APPLY, U.n, my_size, this->num_map );
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	for( int i = 0; i < this->num_map; ++i ){
		MPI_Allgatherv(&tmp_ret(i*my_size,0), size[this->rank], get_typecount(tmp_ret(i*my_size,0)).mpi_type,
					   &ret(i*this->num_unit,0), &size[0], &offset[0], get_typecount(ret(i*this->num_unit,0)).mpi_type, this->inner_world);
	}
#endif
	end = std::chrono::system_clock::now();
	this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	if( use_func ){
		beg = std::chrono::system_clock::now();
		ret = (*this->func)(ret, false);
		end = std::chrono::system_clock::now();
		this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	cnt_flop += this->num_map*U.n*my_size + this->num_map*U.n*my_size*3 + this->num_map*U.n*my_size*5;

	return ret;
}
#endif

template<template<typename> class Mat, typename Real>
void BatchNormalize<Mat, Real>::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

	Matrix<Real> tmp_W = this->W[0], tmp_b = this->b[0];
	ifs.read((char*)&tmp_W(0,0), this->num_map*sizeof(tmp_W(0,0)));
	ifs.read((char*)&tmp_b(0,0), this->num_map*sizeof(tmp_b(0,0)));

	this->W[0] = tmp_W;
	this->b[0] = tmp_b;

	Matrix<Real> tmp_mean = mean, tmp_var = var;
	ifs.read((char*)&tmp_mean(0,0), tmp_mean.m*tmp_mean.n*sizeof(tmp_mean(0,0)));
	ifs.read((char*)&tmp_var(0,0), tmp_var.m*tmp_var.n*sizeof(tmp_var(0,0)));

	mean = tmp_mean;
	var = tmp_var;
}

template<template<typename> class Mat, typename Real>
void BatchNormalize<Mat, Real>::output_W ( const std::string& filename )
{
#ifdef USE_MPI
	if( this->rank == 0 ){
#endif
		std::ofstream ofs(filename, std::ios::binary);

		Matrix<Real> tmp_W = this->W[0], tmp_b = this->b[0];
		ofs.write((char*)&tmp_W(0,0), this->num_map*sizeof(tmp_W(0,0)));
		ofs.write((char*)&tmp_b(0,0), this->num_map*sizeof(tmp_b(0,0)));

		Matrix<Real> tmp_mean = mean, tmp_var = var;
		ofs.write((char*)&tmp_mean(0,0), tmp_mean.m*tmp_mean.n*sizeof(tmp_mean(0,0)));
		ofs.write((char*)&tmp_var(0,0), tmp_var.m*tmp_var.n*sizeof(tmp_var(0,0)));
#ifdef USE_MPI
	}
#endif
}
	
#ifdef USE_MPI
template<template<typename> class Mat, typename Real>
void BatchNormalize<Mat, Real>::param_mix ()
{
	int nprocs;
	MPI_Comm_size(this->outer_world, &nprocs);
	if( this->W.size() == 0 ) return;

	int cnt = this->W[0].n + this->b[0].n;
	int tmp = this->W[0].n;
	std::vector<Real> w(cnt);

	auto tmp_W = this->W[0];
#pragma omp parallel for
	for( int i = 0; i < tmp_W.n; ++i ) w[i] = tmp_W(i,0);

	auto tmp_b = this->b[0];
#pragma omp parallel for
	for( int i = 0; i < tmp_b.n; ++i ) w[tmp+i] = tmp_b(i,0);

	MPI_Allreduce(MPI_IN_PLACE, &w[0], cnt, get_typecount(w[0]).mpi_type, MPI_SUM, this->outer_world);
		
#pragma omp parallel for
	for( int i = 0; i < tmp_W.n; ++i ) tmp_W(i,0) = w[i] / nprocs;
	this->W[0] = tmp_W;
	
#pragma omp parallel for
	for( int i = 0; i < tmp_b.n; ++i ) tmp_b(i,0) = w[tmp + i] / nprocs;
	this->b[0] = tmp_b;
}
#endif

#endif
