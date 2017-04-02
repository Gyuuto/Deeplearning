#ifndef DROPOUT_HPP
#define DROPOUT_HPP

#include <fstream>
#include <random>

#include "Layer.hpp"

template<template<typename> class Mat, typename Real>
class Dropout : public Layer<Mat, Real>
{
private:
	Real dropout_p;
#ifdef USE_GPU
	cl_mem cl_offset;
#endif
	
	std::mt19937 mt;
	std::uniform_real_distribution<Real> d_rand;
	Mat<Real> mask;
public:
	Dropout ( int prev_num_map, int prev_num_unit, Real dropout_p, 
			  const std::shared_ptr<Function<Real>>& f );
	~Dropout ();
	
#ifdef USE_MPI
	void init( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world );
#else
	void init( std::mt19937& m );
#endif
	void finalize();
	
	std::pair<std::vector<Mat<Real>>, std::vector<Mat<Real>>> calc_gradient ( const Mat<Real>& U, const Mat<Real>& delta );

	Matrix<Real> calc_delta ( const Matrix<Real>& U, const Matrix<Real>& delta );
#ifdef USE_GPU
	clMatrix<Real> calc_delta ( const clMatrix<Real>& U, const clMatrix<Real>& delta );
#endif
	void update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db );
	
	Matrix<Real> apply ( const Matrix<Real>& U, bool use_func = true );
#ifdef USE_GPU
	clMatrix<Real> apply ( const clMatrix<Real>& U, bool use_func = true );
#endif

	void set_W( const std::string& filename );
	void output_W ( const std::string& filename );

#ifdef USE_MPI
	void param_mix ();
#endif
};

template<template<typename> class Mat, typename Real>
Dropout<Mat, Real>::Dropout( int prev_num_map, int prev_num_unit, Real dropout_p,
							 const std::shared_ptr<Function<Real>>& f )
{
	this->layer_name = "Dropout";

	this->prev_num_map = this->num_map = prev_num_map;
	this->prev_num_unit = this->num_unit = prev_num_unit;
	this->dropout_p = dropout_p;

	this->t_apply = this->t_delta = this->t_grad = 0.0;
	this->t_apply_init = this->t_apply_gemm = this->t_apply_repl = 0.0;
	this->t_delta_init = this->t_delta_gemm = this->t_delta_repl = 0.0;
	this->t_grad_init = this->t_grad_gemm = this->t_grad_repl = 0.0;

	this->func = f;
	d_rand = std::uniform_real_distribution<Real>(0.0, 1.0);

#ifdef USE_GPU
	cl_int err;
	cl_offset = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
#endif
}

template<template<typename> class Mat, typename Real>
Dropout<Mat, Real>::~Dropout()
{
#ifdef USE_GPU
	clReleaseMemObject(cl_offset);
#endif
}

template<template<typename> class Mat, typename Real>
#ifdef USE_MPI
void Dropout<Mat, Real>::init ( std::mt19937& m, MPI_Comm outer_world, MPI_Comm inner_world )
#else
void Dropout<Mat, Real>::init ( std::mt19937& m )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;
	MPI_Comm_size(inner_world, &(this->nprocs));
	MPI_Comm_rank(inner_world, &(this->rank));
#endif

	int seed = time(NULL);
#ifdef USE_MPI
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, inner_world);
#endif
	mt = std::mt19937(seed);

	Matrix<Real> tmp_mask(this->prev_num_unit, this->prev_num_map);
	for( int i = 0; i < this->prev_num_unit; ++i )
		for( int j = 0; j < this->prev_num_map; ++j )
			tmp_mask(i,j) = this->d_rand(mt) < dropout_p ? 0 : 1;
	mask = tmp_mask;
}

template<template<typename> class Mat, typename Real>
void Dropout<Mat, Real>::finalize ()
{
}

template<template<typename> class Mat, typename Real>
std::pair<std::vector<Mat<Real>>, std::vector<Mat<Real>>> Dropout<Mat, Real>::calc_gradient ( const Mat<Real>& U, const Mat<Real>& delta )
{
	return std::make_pair(std::vector<Mat<Real>>(), std::vector<Mat<Real>>());
}

template<template<typename> class Mat, typename Real>
Matrix<Real> Dropout<Mat, Real>::calc_delta ( const Matrix<Real>& U, const Matrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_unit/this->nprocs*U.n;
	}

	my_size = size[this->rank]/U.n;
	my_offset = offset[this->rank]/U.n;
#endif
	Matrix<Real> nx_delta(this->num_map*this->num_unit, delta.n);

	auto U_diff = (*this->prev_func)(U, true);
#pragma omp parallel
	{
		for( int i = 0; i < this->prev_num_map; ++i ){
#pragma omp for nowait
			for( int j = 0; j < my_size; ++j ){
				for( int k = 0; k < delta.n; ++k )
					nx_delta(i*this->num_unit + my_offset + j, k) = delta(i*this->prev_num_map + my_offset + j, k) * U_diff(my_offset + j, k) * mask(my_offset + j, i);
			}
		}
	}

#ifdef USE_MPI
	for( int i = 0; i < this->prev_num_map; ++i )
		MPI_Allgatherv(MPI_IN_PLACE, size[this->rank], get_typecount(nx_delta(i*this->num_unit,0)).mpi_type,
					   &nx_delta(i*this->num_unit,0), &size[0], &offset[0], get_typecount(nx_delta(i*this->num_unit,0)).mpi_type, this->inner_world);
#endif
	auto end = std::chrono::system_clock::now();

	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return nx_delta;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> Dropout<Mat, Real>::calc_delta ( const clMatrix<Real>& U, const clMatrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U[0].n;
		offset[i] = i*this->num_unit/this->nprocs*U[0].n;
	}

	my_size = size[this->rank]/U[0].n;
	my_offset = offset[this->rank]/U[0].n;
#endif

	clMatrix<Real> nx_delta(this->prev_num_map*this->num_unit, delta.n);
	nx_delta = clMatrix<Real>::hadamard(delta, (*this->prev_func)(U, true));
	cl_device_manager.set_argument( PRG::MULT_VEC_MAT, 0, &nx_delta.v );
	cl_device_manager.set_argument( PRG::MULT_VEC_MAT, 1, &mask.v );
	cl_device_manager.set_argument( PRG::MULT_VEC_MAT, 2, &mask.N );
	// maybe it needs offset of row for mask, but for now I assume only running single GPU.
	cl_device_manager.run_kernel( PRG::MULT_VEC_MAT, this->num_unit, nx_delta.n, this->prev_num_map );
	
#ifdef USE_MPI
	for( int i = 0; i < this->prev_num_map; ++i )
		MPI_Allgatherv(MPI_IN_PLACE, size[rank], MPI_DOUBLE_PRECISION,
					   &nx_delta(i*this->num_unit,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#endif
	auto end = std::chrono::system_clock::now();

	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return nx_delta;
}
#endif

template<template<typename> class Mat, typename Real>
void Dropout<Mat, Real>::update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db )
{
	// WIP, do it opencl?
	Matrix<Real> tmp_mask(this->prev_num_map, this->prev_num_unit);
	for( int i = 0; i < this->prev_num_map; ++i )
		for( int j = 0; j < this->prev_num_unit; ++j )
			tmp_mask(j,i) = d_rand(mt) < dropout_p ? 0 : 1;
	mask = tmp_mask;
}

template<template<typename> class Mat, typename Real>
Matrix<Real> Dropout<Mat, Real>::apply ( const Matrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_unit/this->nprocs*U.n;
	}

	my_size = size[this->rank] / U.n;
	my_offset = offset[this->rank] / U.n;
#endif

	Matrix<Real> ret(this->num_map*this->num_unit, U.n), tmp_ret(this->num_map*my_size, U.n);

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int i = 0; i < this->prev_num_map; ++i ){
#pragma omp for nowait
			for( int j = 0; j < my_size; ++j )
				for( int k = 0; k < U.n; ++k )
					tmp_ret(i*my_size + j,k) = U(i*this->num_unit + my_offset+j,k)*(this->is_learning ? mask(my_offset+j,i) : 1.0 - dropout_p);
		}
	}
	
#ifdef USE_MPI
	for( int i = 0; i < this->num_map; ++i )
		MPI_Allgatherv(&tmp_ret(i*my_size,0), size[this->rank], get_typecount(tmp_ret(i*my_size,0)).mpi_type,
					   &ret(i*this->num_unit,0), &size[0], &offset[0], get_typecount(ret(i*this->num_unit,0)).mpi_type, this->inner_world);
#else
	ret = tmp_ret;
#endif
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	if( use_func ) ret = (*this->func)(ret, false);


	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return ret;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> Dropout<Mat, Real>::apply ( const clMatrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	clMatrix<Real> ret, tmp_ret;
	int my_size = this->num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_unit/this->nprocs*U.n;
	}

	my_size = size[this->rank] / U.n;
	my_offset = offset[this->rank] / U.n;
#endif

	ret = clMatrix<Real>(this->num_map*this->num_unit, U.n);
	tmp_ret = U;
	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	if( this->is_learning ){
		cl_device_manager.set_argument( PRG::MULT_VEC_MAT, 0, &tmp_ret.v );
		cl_device_manager.set_argument( PRG::MULT_VEC_MAT, 1, &mask.v );
		cl_device_manager.set_argument( PRG::MULT_VEC_MAT, 2, &mask.N );
		cl_device_manager.run_kernel( PRG::MULT_VEC_MAT, this->num_unit, tmp_ret.n, this->prev_num_map );
	}
	else tmp_ret *= (1.0 - dropout_p);
	
	if( use_func )
		tmp_ret = (*this->func)(tmp_ret, false);

#ifdef USE_MPI
	for( int i = 0; i < this->num_map; ++i )
		MPI_Allgatherv(&tmp_ret(i*this->num_unit,0), size[rank], MPI_FLOAT,
					   &ret(i*this->num_unit,0), &size[0], &offset[0], MPI_FLOAT, inner_world);
#else
	ret = tmp_ret;
#endif
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return ret;
}
#endif

template<template<typename> class Mat, typename Real>
void Dropout<Mat, Real>::set_W ( const std::string& filename )
{
	
}

template<template<typename> class Mat, typename Real>
void Dropout<Mat, Real>::output_W ( const std::string& filename )
{
	
}

#ifdef USE_MPI
template<template<typename> class Mat, typename Real>
void Dropout<Mat, Real>::param_mix ()
{
	
}
#endif

#endif
