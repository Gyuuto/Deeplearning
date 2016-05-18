#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP

#include <fstream>

#include <random>

#include "Layer.hpp"

class FullyConnected : public Layer
{
private:
#ifdef USE_MPI
	MPI_Comm inner_world;
	int rank, nprocs;
#endif
public:
	FullyConnected ( int prev_num_map, int prev_num_unit, int num_map, int num_unit,
					 const std::shared_ptr<Function>& f );

	void init( std::mt19937& m, MPI_Comm inner_world );
	void finalize();
	
	std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	void update_W ( const std::vector<std::vector<Mat>>& dW );
	
	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );

	void set_W( const std::string& filename );
	void output_W ( const std::string& filename );

	void param_mix ();
};

FullyConnected::FullyConnected( int prev_num_map, int prev_num_unit, int num_map, int num_unit,
								const std::shared_ptr<Function>& f )
{
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->num_map = num_map;
	this->num_unit = num_unit;

	func = f;

	rank = 0; nprocs = 1;
}

void FullyConnected::init ( std::mt19937& m, MPI_Comm inner_world )
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	MPI_Comm_size(inner_world, &nprocs);
	MPI_Comm_rank(inner_world, &rank);
#endif

	int offset = 0, my_size = num_unit;
#ifdef USE_MPI
	// currently, divide by holizontal
	my_size = (rank+1)*num_unit/nprocs - rank*num_unit/nprocs;
	offset = rank*num_unit/nprocs;
#endif
	
	for( int i = 0; i < num_map; ++i ){
		W.emplace_back(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j ){
			W[i][j] = Mat(my_size, 1+prev_num_unit);
		}
	}
	
	const double r = sqrt(6.0/(num_unit + prev_num_unit));
	std::uniform_real_distribution<double> d_rand(-r, r);

	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j ){
			for( int k = 0; k < W[i][j].m; ++k ){
				W[i][j](k, 0) = 0;
				for( int l = 1; l < W[i][j].n; ++l )
					W[i][j](k, l) = d_rand(m);
			}
		}
	}
}

void FullyConnected::finalize ()
{
}

std::vector<std::vector<FullyConnected::Mat>> FullyConnected::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	int offset = 0;
#ifdef USE_MPI
	offset = rank*num_unit/nprocs;
#endif
	
	std::vector<std::vector<Mat>> nabla(num_map);
	for( int i = 0; i < num_map; ++i ){
		nabla[i] = std::vector<Mat>(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j )
			nabla[i][j] = Mat(W[i][j].m, W[i][j].n);
	}

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j ){
			Mat V(U[j].m+1, U[j].n), U_ = (*prev_func)(U[j], false);
			for( int k = 0; k < U[j].n; ++k ){
				V(0,k) = 1.0;
				for( int l = 0; l < U[j].m; ++l ) V(l+1,k) = U_(l, k);
			}

			Mat tmp_delta(W[i][j].m, delta[i].n);
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < delta[i].n; ++l )
					tmp_delta(k,l) = delta[i](k+offset,l);
			nabla[i][j] = tmp_delta*Mat::transpose(V);
		}

	return nabla;
}

std::vector<FullyConnected::Mat> FullyConnected::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	std::vector<Mat> tmp_delta(num_map), tmp(prev_num_map), nx_delta(prev_num_map);
	int offset = 0;
#ifdef USE_MPI
	offset = rank*num_unit/nprocs;
#endif
	
	for( int i = 0; i < num_map; ++i ){
		tmp_delta[i] = Mat(W[0][0].m, delta[i].n);
		for( int j = 0; j < delta[i].n; ++j )
			for( int k = 0; k < W[0][0].m; ++k )
				tmp_delta[i](k, j) = delta[i](offset + k, j);
	}

	int i, j, k;
#pragma omp parallel for default(none) \
	private(i,j) shared(tmp, tmp_delta, delta)
	for( i = 0; i < prev_num_map; ++i ){
		tmp[i] = Mat(W[0][0].n, delta[0].n);
		
		for( j = 0; j < num_map; ++j ){
			tmp[i] += Mat::transpose(W[j][i])*tmp_delta[j];
		}
	}

#ifdef USE_MPI
	for( int i = 0; i < prev_num_map; ++i )
		MPI_Allreduce(MPI_IN_PLACE, &tmp[i](0,0), tmp[i].m*tmp[i].n,
					  MPI_DOUBLE_PRECISION, MPI_SUM, inner_world);
#endif
	
	for( int i = 0; i < prev_num_map; ++i )
		nx_delta[i] = Mat(tmp[0].m-1, tmp[0].n);

	for( int i = 0; i < prev_num_map; ++i ){
		int j, k;
		Mat V(tmp[i].m-1, tmp[i].n), U_ = (*prev_func)(U[i], true);
		for( int j = 0; j < tmp[i].m-1; ++j )
			for( int k = 0; k < tmp[i].n; ++k )
				V(j,k) = tmp[i](j+1,k);

		nx_delta[i] = Mat::hadamard(V, U_);
	}
	return nx_delta;
}

void FullyConnected::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			W[i][j] = W[i][j] + dW[i][j];
}

std::vector<FullyConnected::Mat> FullyConnected::apply ( const std::vector<Mat>& U, bool use_func )
{
	std::vector<Mat> ret(num_map), tmp_ret(num_map);
	std::vector<Mat> V(prev_num_map);
	for( int i = 0; i < prev_num_map; ++i ){
		V[i] = Mat(U[i].m+1, U[i].n);
		for( int j = 0; j < U[i].n; ++j ){
			V[i](0,j) = 1.0;
			for( int k = 0; k < U[i].m; ++k )
				V[i](k+1,j) = U[i](k,j);
		}
	}

	for( int i = 0; i < num_map; ++i ){
		tmp_ret[i] = Mat(W[0][0].m, V[0].n);
		ret[i] = Mat(num_unit, V[0].n);
		for( int j = 0; j < prev_num_map; ++j )
			tmp_ret[i] += W[i][j]*V[j];
	}

#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){
		size[i] = ((i+1)*num_unit/nprocs - i*num_unit/nprocs)*U[0].n;
		offset[i] = i*num_unit/nprocs*U[0].n;
	}

	for( int i = 0; i < num_map; ++i )
		MPI_Allgatherv(&tmp_ret[i](0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &ret[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#endif
	
	if( use_func )
		for( int i = 0; i < num_map; ++i )
			ret[i] = (*func)(ret[i], false);
	
	return ret;
}

std::vector<std::vector<FullyConnected::Vec>> FullyConnected::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
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

void FullyConnected::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j ){
			ifs.read((char*)&W[i][j].m, sizeof(W[i][j].m));
			ifs.read((char*)&W[i][j].n, sizeof(W[i][j].n));
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					ifs.read((char*)&W[i][j](k,l), sizeof(W[i][j](k,l)));
		}
}

void FullyConnected::output_W ( const std::string& filename )
{
	std::ofstream ofs(filename, std::ios::binary);

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j ){
			ofs.write((char*)&W[i][j].m, sizeof(W[i][j].m));
			ofs.write((char*)&W[i][j].n, sizeof(W[i][j].n));
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					ofs.write((char*)&W[i][j](k,l), sizeof(W[i][j](k,l)));
		}
}

void FullyConnected::param_mix ()
{
#ifdef USE_MPI
	int nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	if( W.size() == 0 ) return;

	int cnt = W.size()*W[0].size()*W[0][0].m*W[0][0].n;
	std::vector<double> w(cnt);

	int idx = 0;
	for( int i = 0; i < W.size(); ++i )
		for( int j = 0; j < W[i].size(); ++j )
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					w[idx++] = W[i][j](k,l);
		
	MPI_Allreduce(MPI_IN_PLACE, &w[0], cnt, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD);

	idx = 0;
	for( int i = 0; i < W.size(); ++i )
		for( int j = 0; j < W[i].size(); ++j )
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					W[i][j](k,l) = w[idx++]/nprocs;
#endif
}

#endif
