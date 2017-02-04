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
	
	std::vector<std::vector<Mat<Real>>> calc_gradient ( const std::vector<Mat<Real>>& U, const std::vector<Mat<Real>>& delta );
	std::vector<Mat<Real>> calc_delta ( const std::vector<Mat<Real>>& U, const std::vector<Mat<Real>>& delta );
	void update_W ( const std::vector<std::vector<Mat<Real>>>& dW );
	
	std::vector<Mat<Real>> apply ( const std::vector<Mat<Real>>& U, bool use_func = true );
	// std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );

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
			this->W[i][j] = Mat<Real>(my_size, 1+this->prev_num_unit);
		}
	}

	const Real r = sqrt(6.0/(this->num_unit + this->prev_num_unit));
	std::uniform_real_distribution<Real> d_rand(-r, r);
	for( int i = 0; i < this->num_map; ++i ){
		for( int j = 0; j < this->prev_num_map; ++j ){
			for( int k = 0; k < this->num_unit; ++k ){
				if( offset <= k && k < offset+my_size )
					this->W[i][j](k-offset, 0) = 0;
				for( int l = 0; l < this->prev_num_unit; ++l ){
					double a = d_rand(m);
					if( offset <= k && k < offset+my_size )
						this->W[i][j](k-offset, l+1) = a;
				}
			}
		}
	}
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::finalize ()
{
}

template<template<typename> class Mat, typename Real>
std::vector<std::vector<Mat<Real>>> FullyConnected<Mat, Real>::calc_gradient ( const std::vector<Mat<Real>>& U, const std::vector<Mat<Real>>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	
	std::vector<std::vector<Mat<Real>>> nabla(this->num_map);
	for( int i = 0; i < this->num_map; ++i ){
		nabla[i] = std::vector<Mat<Real>>(this->prev_num_map);
		for( int j = 0; j < this->prev_num_map; ++j )
			nabla[i][j] = Mat<Real>(this->W[i][j].m, this->W[i][j].n);
	}
	auto end = std::chrono::system_clock::now();
	this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	Mat<Real> V(U[0].m+1, U[0].n), tmp_delta(this->W[0][0].m, delta[0].n);
	for( int i = 0; i < this->num_map; ++i )
		for( int j = 0; j < this->prev_num_map; ++j ){
			beg = std::chrono::system_clock::now();
			Mat<Real> U_ = (*this->prev_func)(U[j], false);
			end = std::chrono::system_clock::now();
			this->t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

			beg = std::chrono::system_clock::now();
#pragma omp parallel
			{
#pragma omp for schedule(auto) nowait
				for( int l = 0; l < U_.n; ++l ) V(0,l) = (this->is_use_bias ? 1.0 : 0.0);
#pragma omp for schedule(auto) nowait
				for( int k = 0; k < U_.m; ++k ){
					for( int l = 0; l < U_.n; ++l ) V(k+1,l) = U_(k, l);
				}
			
#pragma omp for schedule(auto) nowait
				for( int k = 0; k < tmp_delta.m; ++k )
					for( int l = 0; l < delta[i].n; ++l )
						tmp_delta(k,l) = delta[i](k + offset,l);
			}
			end = std::chrono::system_clock::now();
			this->t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
			
			beg = std::chrono::system_clock::now();
			nabla[i][j] = tmp_delta*Mat<Real>::transpose(V);
			end = std::chrono::system_clock::now();
			this->t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		}
	this->t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nabla;
}


template<template<typename> class Mat, typename Real>
std::vector<Mat<Real>> FullyConnected<Mat, Real>::calc_delta ( const std::vector<Mat<Real>>& U, const std::vector<Mat<Real>>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0;
#ifdef USE_MPI
	offset = this->rank*this->num_unit/this->nprocs;
#endif
	std::vector<Mat<Real>> tmp_delta(this->num_map, Mat<Real>(this->W[0][0].m, delta[0].n)),
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
		tmp[i] = Mat<Real>::zeros(this->W[0][0].n, tmp_delta[0].n);
		if( this->W[0][0].m != 0 )
			for( int j = 0; j < this->num_map; ++j )
				tmp[i] += Mat<Real>::transpose(this->W[j][i])*tmp_delta[j];
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
		Mat<Real> V(this->W[0][0].n-1, tmp_delta[0].n), U_ = (*this->prev_func)(U[i], true);

#ifdef USE_MPI
		beg = std::chrono::system_clock::now();
		MPI_Status stat;
		MPI_Wait(&req[i], &stat);
		end = std::chrono::system_clock::now();
		this->t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
#endif

#pragma omp parallel for schedule(auto)
		for( int j = 0; j < tmp[i].m-1; ++j )
			for( int k = 0; k < tmp[i].n; ++k )
				V(j,k) = tmp[i](j+1,k);
		nx_delta[i] = Mat<Real>::hadamard(V, U_);
	}
	end = std::chrono::system_clock::now();
	this->t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	this->t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}

template<template<typename> class Mat, typename Real>
void FullyConnected<Mat, Real>::update_W ( const std::vector<std::vector<Mat<Real>>>& dW )
{
	for( int i = 0; i < this->num_map; ++i )
		for( int j = 0; j < this->prev_num_map; ++j )
			this->W[i][j] += dW[i][j];
}

template<template<typename> class Mat, typename Real>
std::vector<Mat<Real>> FullyConnected<Mat, Real>::apply ( const std::vector<Mat<Real>>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	std::vector<Mat<Real>> ret(this->num_map), tmp_ret(this->num_map);
	std::vector<Mat<Real>> V(this->prev_num_map, Mat<Real>(U[0].m+1, U[0].n));

	auto end = std::chrono::system_clock::now();
	this->t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	beg = std::chrono::system_clock::now();
#pragma omp parallel
	{
		for( int i = 0; i < this->prev_num_map; ++i ){
#pragma omp for schedule(auto) nowait
			for( int j = 0; j < U[i].n; ++j ) V[i](0,j) = (this->is_use_bias ? 1.0 : 0.0); // for bias
#pragma omp for schedule(auto) nowait
			for( int j = 0; j < U[i].m; ++j ){
				for( int k = 0; k < U[i].n; ++k )
					V[i](j+1,k) = U[i](j,k);
			}
		}
	}
	end = std::chrono::system_clock::now();
	this->t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

#ifdef USE_MPI
	beg = std::chrono::system_clock::now();
	for( int i = 0; i < this->num_map; ++i ){
		tmp_ret[i] = this->W[i][0]*V[0];
		ret[i] = Mat<Real>(this->num_unit, V[0].n);
		for( int j = 1; j < this->prev_num_map; ++j )
			tmp_ret[i] += this->W[i][j]*V[j];
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
		ret[i] = this->W[i][0]*V[0];
		for( int j = 1; j < this->prev_num_map; ++j )
			ret[i] += this->W[i][j]*V[j];
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

// std::vector<std::vector<FullyConnected::Vec>> FullyConnected::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
// {
// 	std::vector<Mat> tmp(prev_num_map);
// 	for( int i = 0; i < prev_num_map; ++i )
// 		tmp[i] = Mat(u[i][0].size(), u.size());

// #ifdef USE_GPU
// 	puts("WIP");
// #else
// #pragma omp parallel
// 	{
// 		for( int i = 0; i < prev_num_map; ++i )
// #pragma omp for schedule(auto) nowait
// 			for( int j = 0; j < u[i][0].size(); ++j )
// 				for( int k = 0; k < u.size(); ++k )
// 					tmp[i](j,k) = u[k][i][j];
// 	}
// #endif
	
// 	auto U = apply(tmp, use_func);
// 	std::vector<std::vector<Vec>> ret(U[0].n, std::vector<Vec>(U.size(), Vec(U[0].m)));
// #ifdef USE_GPU

// #else
// #pragma omp parallel for schedule(auto)
// 	for( int i = 0; i < U[0].n; ++i ){
// 		for( int j = 0; j < U.size(); ++j )
// 			for( int k = 0; k < U[0].m; ++k )
// 				ret[i][j][k] = U[j](k,i);
// 	}
// #endif

// 	return ret;
// }

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
			for( int k = 0; k < this->W[i][j].m; ++k )
				for( int l = 0; l < this->W[i][j].n; ++l )
					ifs.read((char*)&this->W[i][j](k,l), sizeof(this->W[i][j](k,l)));
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
				Mat<Real> tmp_W = all_W[i][j];
#else
				Mat<Real> tmp_W = this->W[i][j];
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

#ifdef USE_GPU
#include "FullyConnected_gpu.hpp"
#endif

#endif
