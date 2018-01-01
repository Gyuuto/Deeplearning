#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP

#include <fstream>

#include <random>

#include "Layer.hpp"

template<template<typename> class Mat, typename Real>
class FullyConnected : public Layer<Mat, Real>
{
private:
public:
	FullyConnected ( int prev_num_map, int prev_num_unit, int num_map, int num_unit,
					 const std::shared_ptr<Function<Real>>& f, bool use_bias = true );
	~FullyConnected ();
	
#ifdef USE_MPI
	void init( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world );
#else
	void init( std::mt19937& m );
#endif
	void finalize();
	
	void calc_gradient ( const Matrix<Real>& U_apply, const Matrix<Real>& U_diff, const Matrix<Real>& delta, std::vector<Matrix<Real>>& nabla_W, std::vector<Matrix<Real>>& nabla_b );
#ifdef USE_GPU
	void calc_gradient ( const clMatrix<Real>& U_apply, const clMatrix<Real>& U_diff, const clMatrix<Real>& delta, std::vector<clMatrix<Real>>& nabla_W, std::vector<clMatrix<Real>>& nabla_b );
#endif

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

	void set_W( const std::string& filename );
	void output_W ( const std::string& filename );

#ifdef USE_MPI
	void param_mix ();
#endif
};

template<template<typename> class Mat, typename Real>
FullyConnected<Mat, Real>::FullyConnected( int prev_num_map, int prev_num_unit, int num_map, int num_unit,
										   const std::shared_ptr<Function<Real>>& f, bool use_bias )
{
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->num_map = num_map;
	this->num_unit = num_unit;
	this->is_use_bias = use_bias;

#ifdef USE_GPU
	cl_int err, tmp = use_bias;
	this->cl_use_bias = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	err = clEnqueueWriteBuffer( cl_device_manager.get_queue(), this->cl_use_bias, CL_TRUE, 0,
								sizeof(int), &tmp, 0, NULL, NULL );
#endif
	
	this->t_apply = this->t_delta = this->t_grad = 0.0;
	this->t_apply_init = this->t_apply_gemm = this->t_apply_repl = this->t_apply_comm = 0.0;
	this->t_delta_init = this->t_delta_gemm = this->t_delta_repl = this->t_delta_comm = 0.0;
	this->t_grad_init = this->t_grad_gemm = this->t_grad_repl = this->t_grad_comm = 0.0;

	this->func = f;
}

template<template<typename> class Mat, typename Real>
FullyConnected<Mat, Real>::~FullyConnected()
{
#ifdef USE_GPU
	clReleaseMemObject(this->cl_use_bias);
#endif
}

template<template<typename> class Mat, typename Real>
#ifdef USE_MPI
void FullyConnected<Mat, Real>::init ( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world )
#else
void FullyConnected<Mat, Real>::init ( std::mt19937& m )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;
	MPI_Comm_size(inner_world, &(this->nprocs));
	MPI_Comm_rank(inner_world, &(this->rank));

	int offset, my_size;
	// currently, divide by holizontal
	my_size = (this->rank+1)*this->num_map*this->num_unit/this->nprocs - this->rank*this->num_map*this->num_unit/this->nprocs;
	offset = this->rank*this->num_map*this->num_unit/this->nprocs;
#else
	int offset = 0, my_size = this->num_map*this->num_unit;
#endif

	this->W.resize(1);
	this->W[0] = Mat<Real>(my_size, this->prev_num_map*this->prev_num_unit);

	this->b.resize(1);
	this->b[0] = Mat<Real>::zeros(my_size, 1);

	const Real r = sqrt(6.0/(this->num_unit + this->prev_num_unit));
	std::uniform_real_distribution<Real> d_rand(-r, r);
	Matrix<Real> tmp_W = this->W[0];
	for( int i = 0; i < this->num_map*this->num_unit; ++i ){
		for( int j = 0; j < this->prev_num_map*this->prev_num_unit; ++j ){
			double a = d_rand(m);
			if( offset <= i && i < offset+my_size )
				tmp_W(i-offset, j) = a;
		}
	}
	this->W[0] = tmp_W;	
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::finalize ()
{
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::calc_gradient ( const Matrix<Real>& U_apply, const Matrix<Real>& U_diff, const Matrix<Real>& delta, std::vector<Matrix<Real>>& nabla_W, std::vector<Matrix<Real>>& nabla_b )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_map*this->num_unit/this->nprocs;
#endif
	
	Matrix<Real> tmp_delta = delta.sub(offset, 0, this->W[0].m, delta.n);
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	tmp_delta.mult( 1.0, Matrix<Real>::transpose(U_apply), 0.0, nabla_W[0] ); // for future, use view or offset and delete tmp_delta variable
	if( this->is_use_bias ){
#pragma omp parallel for
		for( int i = 0; i < nabla_b[0].m; ++i ){
			nabla_b[0](i, 0) = 0.0;
			for( int j = 0; j < delta.n; ++j ) nabla_b[0](i, 0) += tmp_delta(i, j);
		}
	}
	end = std::chrono::system_clock::now();
	this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::calc_gradient ( const clMatrix<Real>& U_apply, const clMatrix<Real>& U_diff, const clMatrix<Real>& delta, std::vector<clMatrix<Real>>& nabla_W, std::vector<clMatrix<Real>>& nabla_b )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

#ifdef USE_MPI
	int offset = 0;
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	delta.mult( 1.0, clMatrix<Real>::transpose(U_apply), 0.0, nabla_W[0] );
	if( this->is_use_bias ){
		cl_device_manager.set_argument( PRG::FULL_GRAD_BIAS, 0, &nabla_b[0].v );
		cl_device_manager.set_argument( PRG::FULL_GRAD_BIAS, 1, &delta.v );
		cl_device_manager.set_argument( PRG::FULL_GRAD_BIAS, 2, &delta.N );
		cl_device_manager.run_kernel( PRG::FULL_GRAD_BIAS, nabla_b[0].m );
	}
	end = std::chrono::system_clock::now();
	this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}
#endif

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::calc_delta ( const Matrix<Real>& U_apply, const Matrix<Real>& U_diff, const Matrix<Real>& delta, Matrix<Real>& nx_delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	Matrix<Real> tmp_delta;
	tmp_delta = delta.sub(offset, 0, this->W[0].m, delta.n);
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();	
	// nx_delta = Matrix<Real>::transpose(this->W[0])*tmp_delta;
	Matrix<Real>::transpose(this->W[0]).mult( 1.0, tmp_delta, 0.0, nx_delta );
	end = std::chrono::system_clock::now();
	this->t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &nx_delta(0,0), nx_delta.m*nx_delta.n,
				  get_typecount(nx_delta(0,0)).mpi_type, MPI_SUM, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	
	beg = std::chrono::system_clock::now();
	nx_delta.hadamard( U_diff );
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::calc_delta ( const clMatrix<Real>& U_apply, const clMatrix<Real>& U_diff, const clMatrix<Real>& delta, clMatrix<Real>& nx_delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

#ifdef USE_MPI
	int offset = 0;
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	// clMatrix<Real> tmp_delta(this->W[0].m, delta.n), tmp, nx_delta;
	// tmp_delta = delta.sub(offset, 0, this->W[0].m, delta.n);
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	// leave for MPI(multiple GPU), future work
	// beg = std::chrono::system_clock::now();
	// tmp_delta = delta;
	// end = std::chrono::system_clock::now();
	// this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	clMatrix<Real>::transpose(this->W[0]).mult( 1.0, delta, 0.0, nx_delta );
	end = std::chrono::system_clock::now();
	this->t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	MPI_Allreduce(MPI_IN_PLACE, &tmp(0,0), tmp.m*tmp.n,
				  get_typecount(tmp(0,0)).mpi_type, MPI_SUM, this->inner_world);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	nx_delta.hadamard( U_diff );
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}
#endif

template<template<typename> class Mat, typename Real>
Matrix<Real> FullyConnected<Mat, Real>::apply ( const Matrix<Real>& U, bool use_func )
{
	Matrix<Real> ret(this->num_map*this->num_unit, U.n);

	apply( U, ret, use_func );
	
	return ret;
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::apply ( const Matrix<Real>& U, Matrix<Real>& ret, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

#ifdef USE_MPI
	beg = std::chrono::system_clock::now();

	Matrix<Real> tmp_ret(this->W[0].m, U.n);
	this->W[0].mult( 1.0, U, 0.0, tmp_ret );
	if( this->is_use_bias ){
#pragma omp parallel for 
		for( int j = 0; j < tmp_ret.m; ++j )
			for( int k = 0; k < tmp_ret.n; ++k ) tmp_ret(j, k) += this->b[0](j, 0);	
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_map*this->num_unit/this->nprocs - i*this->num_map*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_map*this->num_unit/this->nprocs*U.n;
	}

	MPI_Request req;
	MPI_Iallgatherv(&tmp_ret(0,0), size[this->rank], get_typecount(tmp_ret(0,0)).mpi_type,
					&ret(0,0), &size[0], &offset[0], get_typecount(ret(0,0)).mpi_type, this->inner_world, &req);
	end = std::chrono::system_clock::now();
	this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#else
	beg = std::chrono::system_clock::now();
	// ret = this->W[0]*U; // WIP : to use mult function
	this->W[0].mult( 1.0, U, 0.0, ret );
	if( this->is_use_bias ){
#pragma omp parallel for 
		for( int j = 0; j < ret.m; ++j )
			for( int k = 0; k < ret.n; ++k ) ret(j, k) += this->b[0](j,0);
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

	beg = std::chrono::system_clock::now();
	if( use_func ){	
#ifdef USE_MPI
		beg = std::chrono::system_clock::now();
		MPI_Status stat;
		MPI_Wait(&req, &stat);
		end = std::chrono::system_clock::now();
		this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif
		this->func->inplace(ret, false);
	}
#ifdef USE_MPI
	else{
		beg = std::chrono::system_clock::now();
		MPI_Status stat;
		MPI_Wait(&req, &stat);
		end = std::chrono::system_clock::now();
		this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
#endif
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> FullyConnected<Mat, Real>::apply ( const clMatrix<Real>& U, bool use_func )
{
	clMatrix<Real> ret(this->num_map*this->num_unit, U.n);

	apply( U, ret, use_func );
	
	return ret;
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::apply ( const clMatrix<Real>& U, clMatrix<Real>& ret, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	//// future work : to use multiple GPU ////
	// beg = std::chrono::system_clock::now();
	// for( int i = 0; i < this->num_map; ++i ){
	// 	tmp_ret[i] = this->W[i]*U;
	// 	if( this->is_use_bias ) tmp_ret[i] += this->b[i]*clMatrix<Real>::ones(1, U.n);
	// 	ret[i] = clMatrix<Real>(this->num_unit, U.n);
	// }
	// end = std::chrono::system_clock::now();
	// this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	// beg = std::chrono::system_clock::now();
	// std::vector<int> size(this->nprocs), offset(this->nprocs);
	// for( int i = 0; i < this->nprocs; ++i ){
	// 	size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
	// 	offset[i] = i*this->num_unit/this->nprocs*U.n;
	// }

	// for( int i = 0; i < this->num_map; ++i ){
	// 	Matrix<Real> tmp_ret_ = tmp_ret[i];
	// 	Matrix<Real> ret_ = ret[i];
		
	// 	MPI_Allgatherv(&tmp_ret_[i](0,0), size[this->rank], get_typecount(tmp_ret_[i](0,0)).mpi_type,
	// 					&ret_[i](0,0), &size[0], &offset[0], get_typecount(ret_[i](0,0)).mpi_type, this->inner_world);

	// 	ret[i] = ret_;
	// }
	// end = std::chrono::system_clock::now();
	// this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#else
	auto beg = std::chrono::system_clock::now();
	this->W[0].mult( 1.0, U, 0.0, ret);

	if( this->is_use_bias ){
		cl_device_manager.set_argument( PRG::FULL_APPLY_BIAS, 0, &ret.v );
		cl_device_manager.set_argument( PRG::FULL_APPLY_BIAS, 1, &ret.N );
		cl_device_manager.set_argument( PRG::FULL_APPLY_BIAS, 2, &this->b[0].v );
		cl_device_manager.run_kernel( PRG::FULL_APPLY_BIAS, ret.n, ret.m );
	}
	auto end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

	beg = std::chrono::system_clock::now();
	if( use_func ) this->func->inplace(ret, false);	
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
}
#endif

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db )
{
	this->W[0] += dW[0];

	if( this->is_use_bias ) this->b[0] += db[0];
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

// 	for( int i = 0; i < this->num_map; ++i ){
// 		int my_size = this->W[i].m*this->W[i].n, offset = 0;

// 		Matrix<Real> tmp_W = this->W[i];
// #ifdef USE_MPI
// 		my_size = ((this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs) * this->W[i].n;
// 		offset = this->rank*this->num_unit/this->nprocs * this->W[i].n;
			
// 		ifs.seekg(offset*sizeof(tmp_W(0,0)), std::ios::cur);
// #endif
// 		for( int k = 0; k < tmp_W.m; ++k )
// 			for( int l = 0; l < tmp_W.n; ++l )
// 				ifs.read((char*)&tmp_W(k,l), sizeof(tmp_W(k,l)));
// 		this->W[i] = tmp_W;
// #ifdef USE_MPI
// 		ifs.seekg((this->num_unit*this->W[i].n - (offset + my_size))*sizeof(tmp_W(0,0)), std::ios::cur);
// #endif
		
// 		Matrix<Real> tmp_b = this->b[i];
// 		my_size = this->b[i].m; offset = 0;
// #ifdef USE_MPI
// 		my_size = ((this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs);
// 		offset = this->rank*this->num_unit/this->nprocs;
			
// 		ifs.seekg(offset*sizeof(tmp_b(0,0)), std::ios::cur);
// #endif
// 		for( int k = 0; k < tmp_b.m; ++k ) ifs.read((char*)&tmp_b(k,0), sizeof(tmp_b(k,0)));
// 		this->b[i] = tmp_b;
// #ifdef USE_MPI
// 		ifs.seekg((this->num_unit - (offset + my_size))*sizeof(tmp_b(0,0)), std::ios::cur);
// #endif
// 	}

	Matrix<Real> tmp_W = this->W[0];
#ifdef USE_MPI
	int offset;
	offset = this->rank*this->num_map*this->num_unit/this->nprocs * this->W[0].n;
			
	ifs.seekg(offset*sizeof(tmp_W(0,0)), std::ios::cur);
#endif
	for( int k = 0; k < tmp_W.m; ++k )
		for( int l = 0; l < tmp_W.n; ++l )
			ifs.read((char*)&tmp_W(k,l), sizeof(tmp_W(k,l)));
	this->W[0] = tmp_W;
		
	Matrix<Real> tmp_b = this->b[0];
#ifdef USE_MPI
	offset = this->rank*this->num_map*this->num_unit/this->nprocs;
			
	ifs.seekg(offset*sizeof(tmp_b(0,0)), std::ios::cur);
#endif
	for( int k = 0; k < tmp_b.m; ++k ) ifs.read((char*)&tmp_b(k,0), sizeof(tmp_b(k,0)));
	this->b[0] = tmp_b;
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::output_W ( const std::string& filename )
{
#ifdef USE_MPI
	Matrix<Real> all_W, all_b;
	if( this->rank == 0 ){
		all_W = Matrix<Real>(this->num_map*this->num_unit, this->prev_num_map*this->prev_num_unit);
		all_b = Matrix<Real>(this->num_map*this->num_unit, 1);

		Matrix<Real> tmp_W = this->W[0];
		for( int j = 0; j < tmp_W.m; ++j )
			for( int k = 0; k < tmp_W.n; ++k )
				all_W(j,k) = tmp_W(j, k);

		Matrix<Real> tmp_b = this->b[0];
		for( int j = 0; j < tmp_b.m; ++j )
			all_b(j,0) = tmp_b(j,0);

		for( int n = 1; n < this->nprocs; ++n ){
			int M, N, offset, my_size;
			MPI_Status tmp[256];
			MPI_Recv(&M, 1, MPI_INTEGER, n, MPI_ANY_TAG, this->inner_world, tmp);
			MPI_Recv(&N, 1, MPI_INTEGER, n, MPI_ANY_TAG, this->inner_world, tmp);
			
			my_size = ((n+1)*this->num_map*this->num_unit/this->nprocs - n*this->num_map*this->num_unit/this->nprocs) * N;
			offset = n*this->num_map*this->num_unit/this->nprocs;

			MPI_Recv(&all_W(offset, 0), my_size, get_typecount(all_W(offset, 0)).mpi_type, n, MPI_ANY_TAG, this->inner_world, tmp);
			MPI_Recv(&all_b(offset, 0), my_size/N, get_typecount(all_b(offset, 0)).mpi_type, n, MPI_ANY_TAG, this->inner_world, tmp);
		}
	}
	else{
		int my_size = ((this->rank+1)*this->num_map*this->num_unit/this->nprocs - this->rank*this->num_map*this->num_unit/this->nprocs) * this->W[0].n;
		MPI_Send(&this->W[0].m, 1, MPI_INTEGER, 0, 0, this->inner_world);
		MPI_Send(&this->W[0].n, 1, MPI_INTEGER, 0, 0, this->inner_world);

		Matrix<Real> tmp_W = this->W[0];
		MPI_Send(&tmp_W(0,0), my_size, get_typecount(tmp_W(0, 0)).mpi_type, 0, 0, this->inner_world);
		Matrix<Real> tmp_b = this->b[0];
		MPI_Send(&tmp_b(0,0), my_size/this->W[0].n, get_typecount(tmp_b(0, 0)).mpi_type, 0, 0, this->inner_world);
	}
#endif

#ifdef USE_MPI
	if( this->rank == 0 ){
#endif
		std::ofstream ofs(filename, std::ios::binary);

#ifdef USE_MPI
		Matrix<Real> tmp_W = all_W;
#else
		Matrix<Real> tmp_W = this->W[0];
#endif
		for( int k = 0; k < tmp_W.m; ++k )
			for( int l = 0; l < tmp_W.n; ++l ){
				ofs.write((char*)&tmp_W(k,l), sizeof(tmp_W(k,l)));
			}

#ifdef USE_MPI
		Matrix<Real> tmp_b = all_b;
#else
		Matrix<Real> tmp_b = this->b[0];
#endif
		for( int k = 0; k < tmp_b.m; ++k )
			ofs.write((char*)&tmp_b(k,0), sizeof(tmp_b(k,0)));
#ifdef USE_MPI
	}
#endif
}

#ifdef USE_MPI
template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::param_mix ()
{
	int nprocs;
	MPI_Comm_size(this->outer_world, &nprocs);
	if( this->W.size() == 0 ) return;

	int cnt = this->W.size()*this->W[0].m*this->W[0].n + this->b.size()*this->b[0].m;
	int tmp = this->W.size()*this->W[0].m*this->W[0].n;
	std::vector<Real> w(cnt);

	for( unsigned int i = 0; i < this->W.size(); ++i ){
		Matrix<Real> tmp_W = this->W[i];
#pragma omp parallel for
		for( int j = 0; j < tmp_W.m; ++j )
			for( int k = 0; k < tmp_W.n; ++k ){
				int idx = i*(tmp_W.m*tmp_W.n) + j*(tmp_W.n) + k;
				w[idx] = tmp_W(j,k);
			}
	}

	for( unsigned int i = 0; i < this->b.size(); ++i ){
		Matrix<Real> tmp_b = this->b[i];
#pragma omp parallel for
		for( int j = 0; j < tmp_b.m; ++j ){
			int idx = tmp + i*tmp_b.m + j;
			w[idx] = tmp_b(j,0);
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, &w[0], cnt, get_typecount(w[0]).mpi_type, MPI_SUM, this->outer_world);

	for( unsigned int i = 0; i < this->W.size(); ++i ){
		Matrix<Real> tmp_W = this->W[i];
#pragma omp parallel for
		for( int j = 0; j < tmp_W.m; ++j )
			for( int k = 0; k < tmp_W.n; ++k ){
				int idx = i*(tmp_W.m*tmp_W.n) + j*tmp_W.n + k;
				tmp_W(j,k) = w[idx]/nprocs;
			}
		this->W[i] = tmp_W;
	}

	for( unsigned int i = 0; i < this->b.size(); ++i ){
		Matrix<Real> tmp_b = this->b[i];
#pragma omp parallel for
		for( int j = 0; j < tmp_b.m; ++j ){
			int idx = tmp + i*tmp_b.m + j;
			tmp_b(j,0) = w[idx]/nprocs;
		}
		this->b[i] = tmp_b;
	}
}
#endif

#endif
