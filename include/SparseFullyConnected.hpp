#ifndef SPARSEFULLYCONNECTED_HPP
#define SPARSEFULLYCONNECTED_HPP

#include <fstream>
#include <random>

#include "Layer.hpp"

class SparseFullyConnected : public Layer
{
private:
	const double R_LAMBDA = 0.9;
	double RHO, BETA;
	Mat rho;
public:
	SparseFullyConnected ( int prev_num_map, int prev_num_unit,
						   int num_mp, int num_unit, double RHO, double BETA,
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

SparseFullyConnected::SparseFullyConnected( int prev_num_map, int prev_num_unit,
											int num_map, int num_unit,
											double RHO, double BETA,
											const std::shared_ptr<Function>& f )
{
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->num_map = num_map;
	this->num_unit = num_unit;
	this->RHO = RHO;
	this->BETA = BETA;

	func = f;

	for( int i = 0; i < num_map; ++i ){
		W.emplace_back(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j ){
			W[i][j] = Mat(num_unit, 1+prev_num_unit);
		}
	}
}

#ifdef USE_MPI
void SparseFullyConnected::init( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world )
#else
void SparseFullyConnected::init ( std::mt19937& m )
#endif
{
	const double r = sqrt(6.0/(prev_num_unit + num_unit));
	std::uniform_real_distribution<double> d_rand(-r, r);
	
	rho = Mat::ones(num_unit, num_map);
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j ){
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					W[i][j](k,l) = d_rand(m);
		}
	}
}

void SparseFullyConnected::finalize ()
{
}

std::vector<std::vector<SparseFullyConnected::Mat>> SparseFullyConnected::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	std::vector<std::vector<Mat>> nabla(num_map);
	for( int i = 0; i < num_map; ++i ){
		nabla[i] = std::vector<Mat>(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j )
			nabla[i][j] = Mat::zeros(W[i][j].m, W[i][j].n);
	}

	auto V = apply(U);
	for( int i = 0; i < V.size(); ++i ){
		auto V_ = (*func)(V[i], false);
		for( int j = 0; j < V[i].m; ++j ){
			double tmp_rho = 0.0;
			for( int k = 0; k < V[i].n; ++k )
				tmp_rho += V_(j,k);
			rho(j,i) = R_LAMBDA*rho(j,i) + (1.0-R_LAMBDA)*tmp_rho/V[i].n;
		}
	}

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j ){
			Mat V(U[j].m+1, U[j].n), U_ = (*prev_func)(U[j], false);
			for( int k = 0; k < U[j].n; ++k ){
				V(0,k) = 1.0;
				for( int l = 0; l < U[j].m; ++l ) V(l+1,k) = U_(l, k);
			}
			auto delta_ = delta[i];
			for( int k = 0; k < delta[i].m; ++k ){
				double KL = (1.0-RHO)/(1.0-rho(k,i)) - RHO/rho(k,i);
				for( int l = 0; l < delta[i].n; ++l ) delta_(k,l) += BETA*KL;
			}
					
			nabla[i][j] = delta_*Mat::transpose(V);
		}
	
	return nabla;
}

std::vector<SparseFullyConnected::Mat> SparseFullyConnected::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	std::vector<Mat> tmp(prev_num_map), nx_delta(prev_num_map), delta_(num_map);
	for( int i = 0; i < num_map; ++i ){
		delta_[i] = delta[i];
		for( int j = 0; j < delta_[i].m; ++j )
			for( int k = 0; k < delta_[i].n; ++k )
				delta_[i](j,k) += BETA*((1.0-RHO)/(1.0-rho(j,i)) - RHO/rho(j,i));
	}
	
	for( int i = 0; i < prev_num_map; ++i ){
		tmp[i] = Mat(W[0][0].n, delta[0].n);
		for( int j = 0; j < num_map; ++j )
			tmp[i] = tmp[i] + Mat::transpose(W[j][i])*delta[j];
	}
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

void SparseFullyConnected::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			W[i][j] = W[i][j] + dW[i][j];
}

std::vector<SparseFullyConnected::Mat> SparseFullyConnected::apply ( const std::vector<Mat>& U, bool use_func )
{
	std::vector<Mat> ret(num_map);
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
		ret[i] = Mat(W[i][0].m, V[0].n);
		for( int j = 0; j < prev_num_map; ++j )
			ret[i] = ret[i] + W[i][j]*V[j];
	}

	if( use_func )
		for( int i = 0; i < num_map; ++i )
			ret[i] = (*func)(ret[i], false);
	
	return ret;
}

std::vector<std::vector<SparseFullyConnected::Vec>> SparseFullyConnected::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
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

void SparseFullyConnected::set_W ( const std::string& filename )
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

void SparseFullyConnected::output_W ( const std::string& filename )
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

#ifdef USE_MPI
void SparseFullyConnected::param_mix ()
{
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
}	
#endif

#endif
