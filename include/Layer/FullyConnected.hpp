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
	my_size = (this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs;
	offset = this->rank*this->num_unit/this->nprocs;
#else
	int offset = 0, my_size = this->num_unit;
#endif

	this->W.resize(this->num_map);
	for( int i = 0; i < this->num_map; ++i ){
		this->W[i] = Mat<Real>(my_size, this->prev_num_map*this->prev_num_unit);
	}

	this->b.resize(this->num_map);
	for( int i = 0; i < this->num_map; ++i ){
		this->b[i] = Mat<Real>::zeros(my_size, 1);
	}

	const Real r = sqrt(6.0/(this->num_unit + this->prev_num_unit));
	std::uniform_real_distribution<Real> d_rand(-r, r);
	for( int i = 0; i < this->num_map; ++i ){

		Matrix<Real> tmp_W = this->W[i];
		for( int k = 0; k < this->num_unit; ++k ){
			for( int l = 0; l < this->prev_num_map*this->prev_num_unit; ++l ){
				double a = d_rand(m);
				if( offset <= k && k < offset+my_size )
					tmp_W(k-offset, l) = a;
			}
		}
		this->W[i] = tmp_W;	
	}
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::finalize ()
{
}

template<template<typename> class Mat, typename Real>
std::pair<std::vector<Matrix<Real>>, std::vector<Matrix<Real>>> FullyConnected<Mat, Real>::calc_gradient ( const Matrix<Real>& U, const Matrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	
	std::vector<Matrix<Real>> nabla_W(this->num_map), nabla_b(this->num_map);
	for( int i = 0; i < this->num_map; ++i ){
		nabla_W[i] = Matrix<Real>(this->W[i].m, this->W[i].n);
		nabla_b[i] = Matrix<Real>(this->W[i].m, 1);
	}
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	Matrix<Real> tmp_delta(this->W[0].m, delta.n), e;
	if( this->is_use_bias ) e = Matrix<Real>::ones(delta.n, 1);

	beg = std::chrono::system_clock::now();
	Matrix<Real> U_ = (*this->prev_func)(U, false);
	end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	for( int i = 0; i < this->num_map; ++i ){
		beg = std::chrono::system_clock::now();
#pragma omp parallel for 
		for( int k = 0; k < tmp_delta.m; ++k )
			for( int l = 0; l < delta.n; ++l )
				tmp_delta(k,l) = delta(i*this->num_unit + k + offset,l);
		end = std::chrono::system_clock::now();
		this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
			
		beg = std::chrono::system_clock::now();
		nabla_W[i] = tmp_delta*Matrix<Real>::transpose(U_);
		if( this->is_use_bias ) nabla_b[i] = tmp_delta*e;
		else nabla_b[i] = Matrix<Real>::zeros(tmp_delta.m, 1);
		end = std::chrono::system_clock::now();
		this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return std::make_pair(nabla_W, nabla_b);
}


#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
std::pair<std::vector<clMatrix<Real>>, std::vector<clMatrix<Real>>> FullyConnected<Mat, Real>::calc_gradient ( const clMatrix<Real>& U, const clMatrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	
	std::vector<clMatrix<Real>> nabla_W(this->num_map), nabla_b(this->num_map);
	for( int i = 0; i < this->num_map; ++i ){
		nabla_W[i] = clMatrix<Real>(this->W[i].m, this->W[i].n);
		nabla_b[i] = clMatrix<Real>(this->W[i].m, 1);
	}
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	// clMatrix<Real> V(U[0].m+1, U[0].n), tmp_delta(this->W[0][0].m, delta[0].n), e;
	clMatrix<Real> e;
	if( this->is_use_bias ) e = Matrix<Real>::ones(delta.n, 1);

	beg = std::chrono::system_clock::now();
	clMatrix<Real> U_ = (*this->prev_func)(U, false);
	end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	clMatrix<Real> tmp_delta(this->W[0].m, delta.n);
	for( int i = 0; i < this->num_map; ++i ){
		// beg = std::chrono::system_clock::now();
		// tmp_delta = delta[i];
		// end = std::chrono::system_clock::now();
		// this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		tmp_delta = delta.sub(i*this->num_unit, 0, this->num_unit, delta.n);
		beg = std::chrono::system_clock::now();
		nabla_W[i] = tmp_delta*clMatrix<Real>::transpose(U_);
		if( this->is_use_bias ) nabla_b[i] = tmp_delta*e;
		else nabla_b[i] = Matrix<Real>::zeros(delta.m, 1);
		end = std::chrono::system_clock::now();
		this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		}
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return std::make_pair(nabla_W, nabla_b);
}
#endif

template<template<typename> class Mat, typename Real>
Matrix<Real> FullyConnected<Mat, Real>::calc_delta ( const Matrix<Real>& U, const Matrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	Matrix<Real> tmp_delta(this->W[0].m, delta.n), tmp, nx_delta;
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int i = 0; i < this->num_map; ++i ){
#pragma omp for nowait
			for( int j = 0; j < this->W[i].m; ++j ){
				for( int k = 0; k < delta.n; ++k )
					tmp_delta(i*this->W[i].m + j, k) = delta(i*this->num_unit + offset + j, k);
			}
		}
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	tmp = Matrix<Real>::zeros(this->W[0].n, tmp_delta.n);
	if( this->W[0].m != 0 ){
		for( int i = 0; i < this->num_map; ++i )
			tmp += Matrix<Real>::transpose(this->W[i])*tmp_delta;
	}
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
	nx_delta = Matrix<Real>::hadamard(tmp, (*this->prev_func)(U, true));
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> FullyConnected<Mat, Real>::calc_delta ( const clMatrix<Real>& U, const clMatrix<Real>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	clMatrix<Real> tmp_delta(this->W[0].m, delta.n), tmp, nx_delta;
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	// leave for MPI(multiple GPU), future work
	beg = std::chrono::system_clock::now();
	tmp_delta = delta;
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	tmp = clMatrix<Real>::zeros(this->W[0].n, tmp_delta.n);
	if( this->W[0].m != 0 ){
		for( int i = 0; i < this->num_map; ++i ){
			tmp += clMatrix<Real>::transpose(this->W[i])*tmp_delta;
		}
	}
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
	nx_delta = clMatrix<Real>::hadamard(tmp, (*this->prev_func)(U, true));
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}
#endif

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db )
{
	for( int i = 0; i < this->num_map; ++i ) this->W[i] += dW[i];

	if( this->is_use_bias ){
		for( int i = 0; i < this->num_map; ++i ) this->b[i] += db[i];
	}
}

template<template<typename> class Mat, typename Real>
Matrix<Real> FullyConnected<Mat, Real>::apply ( const Matrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	std::vector<Matrix<Real>> ret(this->num_map), tmp_ret(this->num_map);

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		tmp_ret[i] = this->W[i]*U;
		if( this->is_use_bias ) tmp_ret[i] += this->b[i]*Matrix<Real>::ones(1, U.n);
		ret[i] = Matrix<Real>(this->num_unit, U.n);
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_unit/this->nprocs*U.n;
	}

	std::vector<MPI_Request> req(this->num_map);
	for( int i = 0; i < this->num_map; ++i )
		MPI_Iallgatherv(&tmp_ret[i](0,0), size[this->rank], get_typecount(tmp_ret[i](0,0)).mpi_type,
						&ret[i](0,0), &size[0], &offset[0], get_typecount(ret[i](0,0)).mpi_type, this->inner_world, &req[i]);
	end = std::chrono::system_clock::now();
	this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#else
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		ret[i] = this->W[i]*U;
		if( this->is_use_bias ) ret[i] += this->b[i]*Matrix<Real>::ones(1, U.n);
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

	beg = std::chrono::system_clock::now();
	if( use_func )
		for( int i = 0; i < this->num_map; ++i ){
#ifdef USE_MPI
			beg = std::chrono::system_clock::now();
			MPI_Status stat;
			MPI_Wait(&req[i], &stat);
			end = std::chrono::system_clock::now();
			this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif
			ret[i] = (*this->func)(ret[i], false);
		}
#ifdef USE_MPI
	else{
		beg = std::chrono::system_clock::now();
		for( int i = 0; i < this->num_map; ++i ){
			MPI_Status stat;
			MPI_Wait(&req[i], &stat);
		}
		end = std::chrono::system_clock::now();
		this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
#endif
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return Matrix<Real>::to_matrix(ret);
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
clMatrix<Real> FullyConnected<Mat, Real>::apply ( const clMatrix<Real>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	std::vector<clMatrix<Real>> ret(this->num_map), tmp_ret(this->num_map);

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		tmp_ret[i] = this->W[i]*U;
		if( this->is_use_bias ) tmp_ret[i] += this->b[i]*clMatrix<Real>::ones(1, U.n);
		ret[i] = clMatrix<Real>(this->num_unit, U.n);
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U.n;
		offset[i] = i*this->num_unit/this->nprocs*U.n;
	}

	for( int i = 0; i < this->num_map; ++i ){
		Matrix<Real> tmp_ret_ = tmp_ret[i];
		Matrix<Real> ret_ = ret[i];
		
		MPI_Allgatherv(&tmp_ret_[i](0,0), size[this->rank], get_typecount(tmp_ret_[i](0,0)).mpi_type,
						&ret_[i](0,0), &size[0], &offset[0], get_typecount(ret_[i](0,0)).mpi_type, this->inner_world);

		ret[i] = ret_;
	}
	end = std::chrono::system_clock::now();
	this->t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#else
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		ret[i] = this->W[i]*U;
		if( this->is_use_bias ) ret[i] += this->b[i]*clMatrix<Real>::ones(1, U.n);
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif
	
	beg = std::chrono::system_clock::now();
	if( use_func ){
		for( int i = 0; i < this->num_map; ++i ){
			ret[i] = (*this->func)(ret[i], false);
		}
	}
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	this->t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return clMatrix<Real>::to_matrix(ret);
}
#endif

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

	for( int i = 0; i < this->num_map; ++i ){
		int my_size = this->W[i].m*this->W[i].n, offset = 0;

		Matrix<Real> tmp_W = this->W[i];
#ifdef USE_MPI
		my_size = ((this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs) * this->W[i].n;
		offset = this->rank*this->num_unit/this->nprocs * this->W[i].n;
			
		ifs.seekg(offset*sizeof(tmp_W(0,0)), std::ios::cur);
#endif
		for( int k = 0; k < tmp_W.m; ++k )
			for( int l = 0; l < tmp_W.n; ++l )
				ifs.read((char*)&tmp_W(k,l), sizeof(tmp_W(k,l)));
		this->W[i] = tmp_W;
#ifdef USE_MPI
		ifs.seekg((this->num_unit*this->W[i].n - (offset + my_size))*sizeof(tmp_W(0,0)), std::ios::cur);
#endif
		
		Matrix<Real> tmp_b = this->b[i];
		my_size = this->b[i].m; offset = 0;
#ifdef USE_MPI
		my_size = ((this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs);
		offset = this->rank*this->num_unit/this->nprocs;
			
		ifs.seekg(offset*sizeof(tmp_b(0,0)), std::ios::cur);
#endif
		for( int k = 0; k < tmp_b.m; ++k ) ifs.read((char*)&tmp_b(k,0), sizeof(tmp_b(k,0)));
		this->b[i] = tmp_b;
#ifdef USE_MPI
		ifs.seekg((this->num_unit - (offset + my_size))*sizeof(tmp_b(0,0)), std::ios::cur);
#endif
	}
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::output_W ( const std::string& filename )
{
#ifdef USE_MPI
	std::vector<Matrix<Real>> all_W, all_b;
	if( this->rank == 0 ){
		all_W = std::vector<Matrix<Real>>(this->num_map, Matrix<Real>(this->num_unit, this->prev_num_map*this->prev_num_unit));
		all_b = std::vector<Matrix<Real>>(this->num_map, Matrix<Real>(this->num_unit, 1));

		for( int i = 0; i < this->num_map; ++i ){
			Matrix<Real> tmp_W = this->W[i];
			for( int j = 0; j < tmp_W.m; ++j )
				for( int k = 0; k < tmp_W.n; ++k )
					all_W[i](j,k) = tmp_W(j, k);
		}

		for( int i = 0; i < this->num_map; ++i ){
			Matrix<Real> tmp_b = this->b[i];
			for( int j = 0; j < tmp_b.m; ++j )
				all_b[i](j,0) = tmp_b(j,0);
		}

		for( int n = 1; n < this->nprocs; ++n ){
			int M, N, offset, my_size;
			MPI_Status tmp[256];
			MPI_Recv(&M, 1, MPI_INTEGER, n, MPI_ANY_TAG, this->inner_world, tmp);
			MPI_Recv(&N, 1, MPI_INTEGER, n, MPI_ANY_TAG, this->inner_world, tmp);
			
			my_size = ((n+1)*this->num_unit/this->nprocs - n*this->num_unit/this->nprocs) * N;
			offset = n*this->num_unit/this->nprocs;

			for( int i = 0; i < this->num_map; ++i )
				MPI_Recv(&all_W[i](offset, 0), my_size, get_typecount(all_W[i](offset, 0)).mpi_type, n, MPI_ANY_TAG, this->inner_world, tmp);
			
			for( int i = 0; i < this->num_map; ++i )
				MPI_Recv(&all_b[i](offset, 0), my_size/N, get_typecount(all_b[i](offset, 0)).mpi_type, n, MPI_ANY_TAG, this->inner_world, tmp);
		}
	}
	else{
		int my_size = ((this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs) * this->W[0].n;
		MPI_Send(&this->W[0].m, 1, MPI_INTEGER, 0, 0, this->inner_world);
		MPI_Send(&this->W[0].n, 1, MPI_INTEGER, 0, 0, this->inner_world);

		for( int i = 0; i < this->num_map; ++i ){
			Matrix<Real> tmp_W = this->W[i];
			MPI_Send(&tmp_W(0,0), my_size, get_typecount(tmp_W(0, 0)).mpi_type, 0, 0, this->inner_world);
		}
		for( int i = 0; i < this->num_map; ++i ){
			Matrix<Real> tmp_b = this->b[i];
			MPI_Send(&tmp_b(0,0), my_size/this->W[0].n, get_typecount(tmp_b(0, 0)).mpi_type, 0, 0, this->inner_world);
		}
	}
#endif

#ifdef USE_MPI
	if( this->rank == 0 ){
#endif
		std::ofstream ofs(filename, std::ios::binary);

		for( int i = 0; i < this->num_map; ++i ){
#ifdef USE_MPI
			Matrix<Real> tmp_W = all_W[i];
#else
			Matrix<Real> tmp_W = this->W[i];
#endif
			for( int k = 0; k < this->num_unit; ++k )
				for( int l = 0; l < this->W[i].n; ++l ){
					ofs.write((char*)&tmp_W(k,l), sizeof(tmp_W(k,l)));
				}

#ifdef USE_MPI
			Matrix<Real> tmp_b = all_b[i];
#else
			Matrix<Real> tmp_b = this->b[i];
#endif
			for( int k = 0; k < this->num_unit; ++k )
				ofs.write((char*)&tmp_b(k,0), sizeof(tmp_b(k,0)));
		}
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

	for( int i = 0; i < this->W.size(); ++i ){
		Matrix<Real> tmp_W = this->W[i];
#pragma omp parallel for
		for( int j = 0; j < tmp_W.m; ++j )
			for( int k = 0; k < tmp_W.n; ++k ){
				int idx = i*(tmp_W.m*tmp_W.n) + j*(tmp_W.n) + k;
				w[idx] = tmp_W(j,k);
			}
	}

	for( int i = 0; i < this->b.size(); ++i ){
		Matrix<Real> tmp_b = this->b[i];
#pragma omp parallel for
		for( int j = 0; j < tmp_b.m; ++j ){
			int idx = tmp + i*tmp_b.m + j;
			w[idx] = tmp_b(j,0);
		}
	}

	MPI_Allreduce(MPI_IN_PLACE, &w[0], cnt, get_typecount(w[0]).mpi_type, MPI_SUM, this->outer_world);

	for( int i = 0; i < this->W.size(); ++i ){
		Matrix<Real> tmp_W = this->W[i];
#pragma omp parallel for
		for( int j = 0; j < tmp_W.m; ++j )
			for( int k = 0; k < tmp_W.n; ++k ){
				int idx = i*(tmp_W.m*tmp_W.n) + j*tmp_W.n + k;
				tmp_W(j,k) = w[idx]/nprocs;
			}
		this->W[i] = tmp_W;
	}

	for( int i = 0; i < this->b.size(); ++i ){
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
