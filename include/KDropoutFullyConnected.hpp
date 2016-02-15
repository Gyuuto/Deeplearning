#ifndef KDROPOUTFULLYCONNECTED_HPP
#define KDROPOUTFULLYCONNECTED_HPP

#include <fstream>

#include <random>

#include "Layer.hpp"

class KDropoutFullyConnected : public Layer
{
private:
	int K;

	std::mt19937 mt;
	std::uniform_real_distribution<double> d_rand;
	Mat mask;
public:
	KDropoutFullyConnected ( int prev_num_map, int prev_num_unit, int num_map, int num_unit,
							 int K, 
							 const std::function<double(double)>& f,
							 const std::function<double(double)>& d_f );

	void init( std::mt19937& m );
	void finalize();
	
	std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	void update_W ( const std::vector<std::vector<Mat>>& dW );
	
	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );

	void set_W( const std::string& filename );
	void output_W ( const std::string& filename );
};

KDropoutFullyConnected::KDropoutFullyConnected( int prev_num_map, int prev_num_unit,
											  int num_map, int num_unit,
												int K,
											  const std::function<double(double)>& f,
											  const std::function<double(double)>& d_f )
{
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->num_map = num_map;
	this->num_unit = num_unit;
	this->K = K;

	mt = std::mt19937(time(NULL));
	d_rand = std::uniform_real_distribution<double>(0.0, 1.0);
	
	activate_func = f;
	activate_diff_func = d_f;
}

void KDropoutFullyConnected::init ( std::mt19937& m )
{
	const double r = sqrt(6.0/(num_unit + prev_num_unit));
	std::uniform_real_distribution<double> d_rand(-r, r);
	for( int i = 0; i < num_map; ++i ){
		W.emplace_back(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j ){
			W[i][j] = Mat(num_unit, 1+prev_num_unit);
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					W[i][j](k,l) = d_rand(m);
		}
	}

	mask = Mat(prev_num_unit, prev_num_map);
	for( int i = 0; i < prev_num_unit; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			mask(i,j) = 1;
}

void KDropoutFullyConnected::finalize ()
{
	
}

std::vector<std::vector<KDropoutFullyConnected::Mat>> KDropoutFullyConnected::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	std::vector<std::vector<Mat>> nabla(num_map);
	for( int i = 0; i < num_map; ++i ){
		nabla[i] = std::vector<Mat>(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j )
			nabla[i][j] = Mat(W[i][j].m, W[i][j].n);
	}

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			for( int k = 0; k < nabla[i][j].m; ++k )
				for( int l = 0; l < nabla[i][j].n; ++l )
					for( int m = 0; m < delta[i].n; ++m )
						nabla[i][j](k,l) += delta[i](k,m)*(
							l == 0 ? 1.0 : mask(l-1,j)*prev_activate_func(U[j](l-1,m))
							);

	return nabla;
}

std::vector<KDropoutFullyConnected::Mat> KDropoutFullyConnected::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	std::vector<Mat> tmp(prev_num_map), nx_delta(prev_num_map);

	for( int i = 0; i < prev_num_map; ++i ){
		tmp[i] = Mat(W[0][0].n, delta[0].n);
		for( int j = 0; j < num_map; ++j ){
			tmp[i] = tmp[i] + Mat::transpose(W[j][i])*delta[j];
		}
	}
	for( int i = 0; i < prev_num_map; ++i )
		nx_delta[i] = Mat(tmp[0].m-1, tmp[0].n);

	for( int i = 0; i < prev_num_map; ++i )
		for( int j = 0; j < tmp[i].m-1; ++j )
			for( int k = 0; k < tmp[i].n; ++k )
				nx_delta[i](j,k) += mask(j,i)*tmp[i](j+1,k)*prev_activate_diff_func(U[i](j,k));
	
	return nx_delta;
}

void KDropoutFullyConnected::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j )
			W[i][j] = W[i][j] + dW[i][j];
	}
}

std::vector<KDropoutFullyConnected::Mat> KDropoutFullyConnected::apply ( const std::vector<Mat>& U, bool use_func )
{
	std::vector<Mat> ret(num_map);
	std::vector<Mat> V(prev_num_map);
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

		for( int j = 0; j < K; ++j ) mask(idx[j],i) = 1.0;
		for( int j = K; j < U[i].m; ++j ) mask(idx[j],i) = 0.0;
	}

	for( int i = 0; i < prev_num_map; ++i ){
		V[i] = Mat(U[i].m+1, U[i].n);

		for( int j = 0; j < U[i].n; ++j ) V[i](0,j) = 1.0;
		for( int j = 0; j < U[i].m; ++j ){
			for( int k = 0; k < U[i].n; ++k ){
				V[i](j+1,k) = U[i](j,k)*mask(j,i);
			}
		}
	}

	for( int i = 0; i < num_map; ++i ){
		ret[i] = Mat(W[i][0].m, V[0].n);
		for( int j = 0; j < prev_num_map; ++j ){
			ret[i] = ret[i] + W[i][j]*V[j];
		}
	}

	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < ret[i].m; ++j )
			for( int k = 0; k < ret[i].n; ++k )
				ret[i](j,k) = (use_func ? activate_func(ret[i](j,k)) : ret[i](j,k));
	}
	
	return ret;
}

std::vector<std::vector<KDropoutFullyConnected::Vec>> KDropoutFullyConnected::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
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

void KDropoutFullyConnected::set_W ( const std::string& filename )
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

void KDropoutFullyConnected::output_W ( const std::string& filename )
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

#endif
