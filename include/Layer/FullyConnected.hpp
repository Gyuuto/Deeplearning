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
	
	std::pair<std::vector<std::vector<Matrix<Real>>>, std::vector<std::vector<Matrix<Real>>>> calc_gradient ( const std::vector<Matrix<Real>>& U, const std::vector<Matrix<Real>>& delta );
#ifdef USE_GPU
	std::pair<std::vector<std::vector<clMatrix<Real>>>, std::vector<std::vector<clMatrix<Real>>>> calc_gradient ( const std::vector<clMatrix<Real>>& U, const std::vector<clMatrix<Real>>& delta );
#endif

	std::vector<Matrix<Real>> calc_delta ( const std::vector<Matrix<Real>>& U, const std::vector<Matrix<Real>>& delta );
#ifdef USE_GPU
	std::vector<clMatrix<Real>> calc_delta ( const std::vector<clMatrix<Real>>& U, const std::vector<clMatrix<Real>>& delta );
#endif

	void update_W ( const std::vector<std::vector<Mat<Real>>>& dW, const std::vector<std::vector<Mat<Real>>>& db );
	
	std::vector<Matrix<Real>> apply ( const std::vector<Matrix<Real>>& U, bool use_func = true );
#ifdef USE_GPU
	std::vector<clMatrix<Real>> apply ( const std::vector<clMatrix<Real>>& U, bool use_func = true );
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

	for( int i = 0; i < this->num_map; ++i ){
		this->W.emplace_back(this->prev_num_map);
		for( int j = 0; j < this->prev_num_map; ++j ){
			this->W[i][j] = Mat<Real>(my_size, this->prev_num_unit);
		}
	}
	for( int i = 0; i < this->num_map; ++i ){
		this->b.emplace_back(this->prev_num_map);
		for( int j = 0; j < this->prev_num_map; ++j ){
			this->b[i][j] = Mat<Real>::zeros(my_size, 1);
		}
	}

	const Real r = sqrt(6.0/(this->num_unit + this->prev_num_unit));
	std::uniform_real_distribution<Real> d_rand(-r, r);
	for( int i = 0; i < this->num_map; ++i ){
		for( int j = 0; j < this->prev_num_map; ++j ){
			Matrix<Real> tmp_W = this->W[i][j];

			for( int k = 0; k < this->num_unit; ++k ){
				for( int l = 0; l < this->prev_num_unit; ++l ){
					double a = d_rand(m);
					if( offset <= k && k < offset+my_size )
						tmp_W(k-offset, l) = a;
				}
			}
			this->W[i][j] = tmp_W;	
		}
	}
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::finalize ()
{
}

template<template<typename> class Mat, typename Real>
std::pair<std::vector<std::vector<Matrix<Real>>>, std::vector<std::vector<Matrix<Real>>>> FullyConnected<Mat, Real>::calc_gradient ( const std::vector<Matrix<Real>>& U, const std::vector<Matrix<Real>>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	
	std::vector<std::vector<Matrix<Real>>> nabla_w(this->num_map), nabla_b(this->num_map);
	for( int i = 0; i < this->num_map; ++i ){
		nabla_w[i] = std::vector<Matrix<Real>>(this->prev_num_map);
		nabla_b[i] = std::vector<Matrix<Real>>(this->prev_num_map);
		for( int j = 0; j < this->prev_num_map; ++j ){
			nabla_w[i][j] = Matrix<Real>(this->W[i][j].m, this->W[i][j].n);
			nabla_b[i][j] = Matrix<Real>(this->W[i][j].m, 1);
		}
	}
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	Matrix<Real> V(U[0].m+1, U[0].n), tmp_delta(this->W[0][0].m, delta[0].n), e;
	if( this->is_use_bias ) e = Matrix<Real>::ones(delta[0].n, 1);
	for( int i = 0; i < this->num_map; ++i )
		for( int j = 0; j < this->prev_num_map; ++j ){
			beg = std::chrono::system_clock::now();
			Matrix<Real> U_ = (*this->prev_func)(U[j], false);
			end = std::chrono::system_clock::now();
			this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

			beg = std::chrono::system_clock::now();
#pragma omp parallel for schedule(auto)
			for( int k = 0; k < tmp_delta.m; ++k )
				for( int l = 0; l < delta[i].n; ++l )
					tmp_delta(k,l) = delta[i](k + offset,l);
			end = std::chrono::system_clock::now();
			this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
			
			beg = std::chrono::system_clock::now();
			nabla_w[i][j] = tmp_delta*Matrix<Real>::transpose(U_);
			if( this->is_use_bias ) nabla_b[i][j] = tmp_delta*e;
			else nabla_b[i][j] = Matrix<Real>::zeros(tmp_delta.m, 1);
			end = std::chrono::system_clock::now();
			this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		}
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return std::make_pair(nabla_w, nabla_b);
}


#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
std::pair<std::vector<std::vector<clMatrix<Real>>>, std::vector<std::vector<clMatrix<Real>>>> FullyConnected<Mat, Real>::calc_gradient ( const std::vector<clMatrix<Real>>& U, const std::vector<clMatrix<Real>>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	
	std::vector<std::vector<clMatrix<Real>>> nabla_w(this->num_map), nabla_b(this->num_map);
	for( int i = 0; i < this->num_map; ++i ){
		nabla_w[i] = std::vector<clMatrix<Real>>(this->prev_num_map);
		nabla_b[i] = std::vector<clMatrix<Real>>(this->prev_num_map);
		for( int j = 0; j < this->prev_num_map; ++j ){
			nabla_w[i][j] = clMatrix<Real>(this->W[i][j].m, this->W[i][j].n);
			nabla_b[i][j] = clMatrix<Real>(this->W[i][j].m, 1);
		}
	}
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	// clMatrix<Real> V(U[0].m+1, U[0].n), tmp_delta(this->W[0][0].m, delta[0].n), e;
	clMatrix<Real> e;
	if( this->is_use_bias ) e = Matrix<Real>::ones(delta[0].n, 1);
	for( int i = 0; i < this->num_map; ++i )
		for( int j = 0; j < this->prev_num_map; ++j ){
			beg = std::chrono::system_clock::now();
			clMatrix<Real> U_ = (*this->prev_func)(U[j], false);
			end = std::chrono::system_clock::now();
			this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

			// beg = std::chrono::system_clock::now();
			// tmp_delta = delta[i];
			// end = std::chrono::system_clock::now();
			// this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
			
			beg = std::chrono::system_clock::now();
			nabla_w[i][j] = delta[i]*clMatrix<Real>::transpose(U_);
			if( this->is_use_bias ) nabla_b[i][j] = delta[i]*e;
			else nabla_b[i][j] = Matrix<Real>::zeros(delta[i].m, 1);
			end = std::chrono::system_clock::now();
			this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		}
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return std::make_pair(nabla_w, nabla_b);
}
#endif

template<template<typename> class Mat, typename Real>
std::vector<Matrix<Real>> FullyConnected<Mat, Real>::calc_delta ( const std::vector<Matrix<Real>>& U, const std::vector<Matrix<Real>>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	std::vector<Matrix<Real>> tmp_delta(this->num_map, Matrix<Real>(this->W[0][0].m, delta[0].n)),
		tmp(this->prev_num_map), nx_delta(this->prev_num_map);
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int i = 0; i < this->num_map; ++i ){
#pragma omp for schedule(auto) nowait
			for( int j = 0; j < this->W[0][0].m; ++j ){
				for( int k = 0; k < delta[i].n; ++k )
					tmp_delta[i](j, k) = delta[i](offset + j, k);
			}
		}
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->prev_num_map; ++i ){
		tmp[i] = Matrix<Real>::zeros(this->W[0][0].n, tmp_delta[0].n);
		if( this->W[0][0].m != 0 )
			for( int j = 0; j < this->num_map; ++j )
				tmp[i] += Matrix<Real>::transpose(this->W[j][i])*tmp_delta[j];
	}
	end = std::chrono::system_clock::now();
	this->t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	std::vector<MPI_Request> req(this->prev_num_map);
	for( int i = 0; i < this->prev_num_map; ++i )
		MPI_Iallreduce(MPI_IN_PLACE, &tmp[i](0,0), tmp[i].m*tmp[i].n,
					   get_typecount(tmp[i](0,0)).mpi_type, MPI_SUM, this->inner_world, &req[i]);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->prev_num_map; ++i ){
		Matrix<Real> U_ = (*this->prev_func)(U[i], true);

#ifdef USE_MPI
		beg = std::chrono::system_clock::now();
		MPI_Status stat;
		MPI_Wait(&req[i], &stat);
		end = std::chrono::system_clock::now();
		this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif
		nx_delta[i] = Matrix<Real>::hadamard(tmp[i], U_);
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
std::vector<clMatrix<Real>> FullyConnected<Mat, Real>::calc_delta ( const std::vector<clMatrix<Real>>& U, const std::vector<clMatrix<Real>>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	std::vector<clMatrix<Real>> tmp_delta(this->num_map, clMatrix<Real>(this->W[0][0].m, delta[0].n)),
		tmp(this->prev_num_map), nx_delta(this->prev_num_map);
	auto end = std::chrono::system_clock::now();
	this->t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	// leave for MPI(multiple GPU), future work
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		tmp_delta[i] = delta[i];
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->prev_num_map; ++i ){
		tmp[i] = clMatrix<Real>::zeros(this->W[0][0].n, tmp_delta[0].n);
		if( this->W[0][0].m != 0 )
			for( int j = 0; j < this->num_map; ++j )
				tmp[i] += clMatrix<Real>::transpose(this->W[j][i])*tmp_delta[j];
	}
	end = std::chrono::system_clock::now();
	this->t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#ifdef USE_MPI
	std::vector<MPI_Request> req(this->prev_num_map);
	for( int i = 0; i < this->prev_num_map; ++i )
		MPI_Iallreduce(MPI_IN_PLACE, &tmp[i](0,0), tmp[i].m*tmp[i].n,
					   get_typecount(tmp[i](0,0)).mpi_type, MPI_SUM, this->inner_world, &req[i]);
#endif
	end = std::chrono::system_clock::now();
	this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->prev_num_map; ++i ){
		clMatrix<Real> U_ = (*this->prev_func)(U[i], true);

#ifdef USE_MPI
		beg = std::chrono::system_clock::now();
		MPI_Status stat;
		MPI_Wait(&req[i], &stat);
		end = std::chrono::system_clock::now();
		this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif
		nx_delta[i] = clMatrix<Real>::hadamard(tmp[i], U_);
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}
#endif

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::update_W ( const std::vector<std::vector<Mat<Real>>>& dW, const std::vector<std::vector<Mat<Real>>>& db )
{
	for( int i = 0; i < this->num_map; ++i )
		for( int j = 0; j < this->prev_num_map; ++j )
			this->W[i][j] += dW[i][j];

	for( int i = 0; i < this->num_map; ++i )
		for( int j = 0; j < this->prev_num_map; ++j )
			this->b[i][j] += db[i][j];
}

template<template<typename> class Mat, typename Real>
std::vector<Matrix<Real>> FullyConnected<Mat, Real>::apply ( const std::vector<Matrix<Real>>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	std::vector<Matrix<Real>> ret(this->num_map), tmp_ret(this->num_map);

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		tmp_ret[i] = this->W[i][0]*U[0];
		if( this->is_use_bias ) tmp_ret[i] += this->b[i][0];
		ret[i] = Matrix<Real>(this->num_unit, U[0].n);
		for( int j = 1; j < this->prev_num_map; ++j ){
			tmp_ret[i] += this->W[i][j]*U[j];
			if( this->is_use_bias ) tmp_ret[i] += this->b[i][j];
		}
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U[0].n;
		offset[i] = i*this->num_unit/this->nprocs*U[0].n;
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
		ret[i] = this->W[i][0]*U[0];
		if( this->is_use_bias ) ret[i] += this->b[i][0]*Matrix<Real>::ones(1, U[0].n);
		for( int j = 1; j < this->prev_num_map; ++j ){
			ret[i] += this->W[i][j]*U[j];
			if( this->is_use_bias ) ret[i] += this->b[i][j]*Matrix<Real>::ones(1, U[j].n);
		}
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
	
	return ret;
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
std::vector<clMatrix<Real>> FullyConnected<Mat, Real>::apply ( const std::vector<clMatrix<Real>>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	std::vector<clMatrix<Real>> ret(this->num_map), tmp_ret(this->num_map);

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		tmp_ret[i] = this->W[i][0]*U[0]:
		if( this->is_use_bias ) tmp_ret[i] += this->b[i][0];
		ret[i] = clMatrix<Real>(this->num_unit, V[0].n);
		for( int j = 1; j < this->prev_num_map; ++j ){
			tmp_ret[i] += this->W[i][j]*U[j];
			if( this->is_use_bias ) tmp_ret[i] += this->b[i][j];
		}
	}
	end = std::chrono::system_clock::now();
	this->t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
	std::vector<int> size(this->nprocs), offset(this->nprocs);
	for( int i = 0; i < this->nprocs; ++i ){
		size[i] = ((i+1)*this->num_unit/this->nprocs - i*this->num_unit/this->nprocs)*U[0].n;
		offset[i] = i*this->num_unit/this->nprocs*U[0].n;
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
		ret[i] = this->W[i][0]*U[0];
		if( this->is_use_bias ) ret[i] += this->b[i][0]*clMatrix<Real>::ones(1, U[0].n);
		for( int j = 1; j < this->prev_num_map; ++j ){
			ret[i] += this->W[i][j]*U[j];
			if( this->is_use_bias ) ret[i] += this->b[i][j]*clMatrix<Real>::ones(1, U[j].n);
		}
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
	
	return ret;
}
#endif

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

	for( int i = 0; i < this->num_map; ++i )
		for( int j = 0; j < this->prev_num_map; ++j ){
			int m, n;
			ifs.read((char*)&m, sizeof(m));
			ifs.read((char*)&n, sizeof(n));

			int my_size = this->W[i][j].m*this->W[i][j].n, offset = 0;
#ifdef USE_MPI
			my_size = ((this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs) * this->W[i][j].n;
			offset = this->rank*this->num_unit/this->nprocs * this->W[i][j].n;
			
			ifs.seekg(offset*sizeof(double), std::ios::cur);

#endif
			Matrix<Real> tmp_W = this->W[i][j];
			for( int k = 0; k < tmp_W.m; ++k )
				for( int l = 0; l < tmp_W.n; ++l )
					ifs.read((char*)&tmp_W(k,l), sizeof(tmp_W(k,l)));
			this->W[i][j] = tmp_W;
#ifdef USE_MPI
			ifs.seekg((this->num_unit * this->W[i][j].n - (offset + my_size))*sizeof(double), std::ios::cur);
#endif
		}
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::output_W ( const std::string& filename )
{
#ifdef USE_MPI
	std::vector<std::vector<Mat<Real>>> all_W;
	if( this->rank == 0 ){
		all_W = std::vector<std::vector<Mat<Real>>>(this->num_map, std::vector<Mat<Real>>(this->prev_num_map, Mat<Real>(this->num_unit, this->prev_num_unit+1)));

		for( int i = 0; i < this->num_map; ++i )
			for( int j = 0; j < this->prev_num_map; ++j )
				for( int k = 0; k < this->W[i][j].m; ++k )
					for( int l = 0; l < this->W[i][j].n; ++l )
						all_W[i][j](k,l) = this->W[i][j](k,l);
		

		for( int n = 1; n < this->nprocs; ++n ){
			int M, N, offset, my_size;
			MPI_Status tmp[256];
			MPI_Recv(&M, 1, MPI_INTEGER, n, MPI_ANY_TAG, this->inner_world, tmp);
			MPI_Recv(&N, 1, MPI_INTEGER, n, MPI_ANY_TAG, this->inner_world, tmp);
			
			my_size = ((n+1)*this->num_unit/this->nprocs - n*this->num_unit/this->nprocs) * N;
			offset = n*this->num_unit/this->nprocs;

			for( int i = 0; i < this->num_map; ++i )
				for( int j = 0; j < this->prev_num_map; ++j )
					MPI_Recv(&all_W[i][j](offset, 0), my_size, get_typecount(all_W[i][j](offset, 0)).mpi_type, n, MPI_ANY_TAG, this->inner_world, tmp);
		}
	}
	else{
		int my_size = ((this->rank+1)*this->num_unit/this->nprocs - this->rank*this->num_unit/this->nprocs) * this->W[0][0].n;
		MPI_Send(&this->W[0][0].m, 1, MPI_INTEGER, 0, 0, this->inner_world);
		MPI_Send(&this->W[0][0].n, 1, MPI_INTEGER, 0, 0, this->inner_world);

		for( int i = 0; i < this->num_map; ++i )
			for( int j = 0; j < this->prev_num_map; ++j )
				MPI_Send(&this->W[i][j](0,0), my_size, get_typecount(this->W[i][j](0, 0)).mpi_type, 0, 0, this->inner_world);
	}
#endif

#ifdef USE_MPI
	if( this->rank == 0 ){
#endif
		std::ofstream ofs(filename, std::ios::binary);

		for( int i = 0; i < this->num_map; ++i )
			for( int j = 0; j < this->prev_num_map; ++j ){
				ofs.write((char*)&this->num_unit, sizeof(this->num_unit));
				ofs.write((char*)&this->W[i][j].n, sizeof(this->W[i][j].n));

#ifdef USE_MPI
				Matrix<Real> tmp_W = all_W[i][j];
#else
				Matrix<Real> tmp_W = this->W[i][j];
#endif
				for( int k = 0; k < this->num_unit; ++k )
					for( int l = 0; l < this->W[i][j].n; ++l ){
						ofs.write((char*)&tmp_W(k,l), sizeof(tmp_W(k,l)));
					}
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

	int cnt = this->W.size()*this->W[0].size()*this->W[0][0].m*this->W[0][0].n;
	std::vector<Real> w(cnt);

#pragma omp parallel
	{
		for( int i = 0; i < this->W.size(); ++i )
			for( int j = 0; j < this->W[i].size(); ++j )
#pragma omp for schedule(auto) nowait
				for( int k = 0; k < this->W[i][j].m; ++k )
					for( int l = 0; l < this->W[i][j].n; ++l ){
						int idx = i*(this->W[i].size()*this->W[i][j].m*this->W[i][j].n) + j*(this->W[i][j].m*this->W[i][j].n) + k*this->W[i][j].n + l;
						w[idx] = this->W[i][j](k,l);
					}
	}

	MPI_Allreduce(MPI_IN_PLACE, &w[0], cnt, get_typecount(w[0]).mpi_type, MPI_SUM, this->outer_world);

#pragma omp parallel
	{
		for( int i = 0; i < this->W.size(); ++i )
			for( int j = 0; j < this->W[i].size(); ++j )
#pragma omp for schedule(auto) nowait
				for( int k = 0; k < this->W[i][j].m; ++k )
					for( int l = 0; l < this->W[i][j].n; ++l ){
						int idx = i*(this->W[i].size()*this->W[i][j].m*this->W[i][j].n) + j*(this->W[i][j].m*this->W[i][j].n) + k*this->W[i][j].n + l;
						this->W[i][j](k,l) = w[idx]/nprocs;
					}
	}
}
#endif

#endif
