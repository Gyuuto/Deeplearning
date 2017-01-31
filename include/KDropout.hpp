#ifndef KDROPOUT_HPP
#define KDROPOUT_HPP

#include <fstream>

#include <random>

#include "Layer.hpp"

class KDropout : public Layer
{
private:
	double K;

	std::mt19937 mt;
	std::uniform_real_distribution<double> d_rand;
	Mat mask;
public:
	KDropout ( int prev_num_map, int prev_num_unit, double K, 
			   const std::shared_ptr<Function>& f );

#ifdef USE_MPI
	void init( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world );
#else
	void init( std::mt19937& m );
#endif
	void finalize();
	
	std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	void update_W ( const std::vector<std::vector<Mat>>& dW );
	
	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );

	void set_W( const std::string& filename );
	void output_W ( const std::string& filename );

#ifdef USE_MPI
	void param_mix ();
#endif
};

KDropout::KDropout( int prev_num_map, int prev_num_unit, double K,
					const std::shared_ptr<Function>& f )
{
	this->prev_num_map = this->num_map = prev_num_map;
	this->prev_num_unit = this->num_unit = prev_num_unit;
	this->K = K;

	t_apply = t_delta = t_grad = 0.0;
	t_apply_init = t_apply_gemm = t_apply_repl = 0.0;
	t_delta_init = t_delta_gemm = t_delta_repl = 0.0;
	t_grad_init = t_grad_gemm = t_grad_repl = 0.0;

	d_rand = std::uniform_real_distribution<double>(0.0, 1.0);
	
	func = f;
}

#ifdef USE_MPI
void KDropout::init ( std::mt19937& m, MPI_Comm outer_world, MPI_Comm inner_world )
#else
void KDropout::init ( std::mt19937& m )
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

	mask = Mat(prev_num_unit, prev_num_map);
	for( int i = 0; i < prev_num_unit; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			mask(i,j) = 1;
}

void KDropout::finalize ()
{
}

std::vector<std::vector<KDropout::Mat>> KDropout::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	return std::vector<std::vector<Mat>>();
}

std::vector<KDropout::Mat> KDropout::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){
		size[i] = ((i+1)*num_unit/nprocs - i*num_unit/nprocs)*U[0].n;
		offset[i] = i*num_unit/nprocs*U[0].n;
	}

	my_size = size[rank]/U[0].n;
	my_offset = offset[rank]/U[0].n;
#endif
	std::vector<Mat> nx_delta(prev_num_map);

#pragma omp parallel
	{
		for( int i = 0; i < num_map; ++i ){
			auto U_diff = (*prev_func)(U[i], true);
			nx_delta[i] = Mat(num_unit, delta[i].n);

#pragma omp for schedule(auto) nowait
			for( int j = 0; j < my_size; ++j ){
				for( int k = 0; k < delta[i].n; ++k )
					nx_delta[i](my_offset+j, k) = delta[i](my_offset + j, k) * U_diff(my_offset + j, k) * mask(j,i);
			}
		}
	}

#ifdef USE_MPI
	for( int i = 0; i < prev_num_map; ++i )
		MPI_Allgatherv(MPI_IN_PLACE, size[rank], MPI_DOUBLE_PRECISION,
					   &nx_delta[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#endif
	auto end = std::chrono::system_clock::now();

	t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return nx_delta;
}

void KDropout::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	
}

std::vector<KDropout::Mat> KDropout::apply ( const std::vector<Mat>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	std::vector<Mat> ret(num_map), tmp_ret(num_map);
	int my_size = num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){
		size[i] = ((i+1)*num_unit/nprocs - i*num_unit/nprocs)*U[0].n;
		offset[i] = i*num_unit/nprocs*U[0].n;
	}

	my_size = size[rank] / U[0].n;
	my_offset = offset[rank] / U[0].n;
#endif
	auto end = std::chrono::system_clock::now();
	t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	for( int i = 0; i < prev_num_map; ++i ){
		std::vector<int> idx(U[i].m);
		std::vector<double> val(U[i].m, 0.0);
		std::iota(idx.begin(), idx.end(), 0);

		for( int j = 0; j < U[i].m; ++j ){
			for( int k = 0; k < U[i].n; ++k ){
				val[j] += U[i](j,k);
			}
			val[j] /= U[i].n;
		}

		std::sort( idx.begin(), idx.end(), [&](int id1, int id2) -> bool {
				return val[id1] > val[id2];
			});

		for( int j = 0; j < prev_num_unit*K; ++j ) mask(idx[j],i) = 1.0;
		for( int j = prev_num_unit*K; j < U[i].m; ++j ) mask(idx[j],i) = 0.0;
	}

#pragma omp parallel
	{
		for( int i = 0; i < prev_num_map; ++i ){
			ret[i] = Mat(num_unit, U[i].n);
			tmp_ret[i] = Mat(my_size, U[i].n);
#pragma omp for schedule(auto) nowait
			for( int j = 0; j < my_size; ++j ){
				for( int k = 0; k < U[i].n; ++k ){
					tmp_ret[i](j,k) = U[i](my_offset+j,k)*mask(my_offset+j,i);
				}
			}
		}
	}

	if( use_func )
		for( int i = 0; i < num_map; ++i )
			tmp_ret[i] = (*func)(tmp_ret[i], false);

#ifdef USE_MPI
	for( int i = 0; i < num_map; ++i )
		MPI_Allgatherv(&tmp_ret[i](0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &ret[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#endif
	end = std::chrono::system_clock::now();
	t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return ret;
}

std::vector<std::vector<KDropout::Vec>> KDropout::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
{
	std::vector<Mat> tmp(prev_num_map);
	for( int i = 0; i < prev_num_map; ++i )
		tmp[i] = Mat(u[i][0].size(), u.size());

	for( int i = 0; i < prev_num_map; ++i )
		for( int j = 0; j < u[i][0].size(); ++j )
			for( int k = 0; k < u.size(); ++k )
				tmp[i](j,k) = u[k][i][j];
	
	auto U = apply(tmp, use_func);
	std::vector<std::vector<Vec>> ret(U[0].n);
	for( int i = 0; i < U[0].n; ++i ){
		ret[i] = std::vector<Vec>(U.size(), Vec(U[0].m));
		for( int j = 0; j < U.size(); ++j )
			for( int k = 0; k < U[0].m; ++k )
				ret[i][j][k] = U[j](k,i);
	}

	return ret;
}

void KDropout::set_W ( const std::string& filename )
{
	
}

void KDropout::output_W ( const std::string& filename )
{

}

#ifdef USE_MPI
void KDropout::param_mix ()
{
	
}	
#endif

#endif
