#ifndef SPARSEFULLYCONNECTED_HPP
#define SPARSEFULLYCONNECTED_HPP

#include "Layer.hpp"

class SparseFullyConnected : public Layer
{
private:
	const double R_LAMBDA = 0.9;
	double RHO, BETA;
	Mat rho;
public:
	SparseFullyConnected ( int num_map, int num_input, int num_output, double RHO, double BETA,
						   const std::function<double(double)>& f,
						   const std::function<double(double)>& d_f );

	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	std::vector<Mat> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	void update_W ( const std::vector<Mat>& dW );

	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );

	void set_W( const std::string& filename );
	void output_W ( const std::string& filename );
};

SparseFullyConnected::SparseFullyConnected( int num_map, int num_input, int num_output,
											double RHO, double BETA,
											const std::function<double(double)>& f,
											const std::function<double(double)>& d_f )
{
	this->num_map = num_map;
	this->num_input = num_input;
	this->num_output = num_output;
	this->RHO = RHO;
	this->BETA = BETA;

	const double r = sqrt(6.0/(num_input + num_output));

	std::mt19937 m(time(NULL));
	std::uniform_real_distribution<double> d_rand(-r, r);
	
	rho = Mat::ones(num_input, num_map);
	for( int i = 0; i < num_map; ++i ){
		W.emplace_back(num_output, 1+num_input);
		for( int j = 0; j < W[i].m; ++j )
			for( int k = 0; k < W[i].n; ++k )
				W[i][j][k] = d_rand(m);
	}

	activate_func = f;
	activate_diff_func = d_f;
}

std::vector<SparseFullyConnected::Mat> SparseFullyConnected::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	std::vector<Mat> tmp(num_map), nx_delta(num_map);
	for( int i = 0; i < num_map; ++i ){
		tmp[i] = Mat::transpose(W[i])*delta[i];
		nx_delta[i] = Mat(tmp[i].m-1, tmp[i].n);
	}

	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < U[i].m; ++j ){
			double tmp_rho = 0.0;
			for( int k = 0; k < U[i].n; ++k )
				tmp_rho += prev_activate_func(U[i][j][k]);
			rho[j][i] = R_LAMBDA*rho[j][i] + (1.0-R_LAMBDA)*tmp_rho;
		}
	}
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < tmp[i].m-1; ++j ){
			double KL = (1.0-RHO)/(1.0-rho[j][i]) - RHO/rho[j][i];
			for( int k = 0; k < tmp[i].n; ++k )
				nx_delta[i][j][k] = (tmp[i][j+1][k]+BETA*KL)*prev_activate_diff_func(U[i][j][k]);
		}
	
	return nx_delta;
}

std::vector<SparseFullyConnected::Mat> SparseFullyConnected::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	std::vector<Mat> nabla(num_map);
	for( int i = 0; i < num_map; ++i )
		nabla[i] = Mat::zeros(W[i].m, W[i].n);

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < nabla[i].m; ++j )
			for( int k = 0; k < nabla[i].n; ++k )
				for( int l = 0; l < delta[i].n; ++l )
					nabla[i][j][k] += delta[i][j][l]*(
						k == 0 ? 1.0 : prev_activate_func(U[i][k-1][l])
						);
	
	return nabla;
}

void SparseFullyConnected::update_W ( const std::vector<Mat>& dW )
{
	for( int i = 0; i < num_map; ++i )
		W[i] = W[i] + dW[i];
}

std::vector<SparseFullyConnected::Mat> SparseFullyConnected::apply ( const std::vector<Mat>& U, bool use_func )
{
	std::vector<Mat> ret(num_map);
	std::vector<Mat> V(num_map);
	for( int i = 0; i < num_map; ++i ){
		V[i] = Mat(U[i].m+1, U[i].n);
		for( int j = 0; j < U[i].n; ++j ){
			V[i][0][j] = 1.0;
			for( int k = 0; k < U[i].m; ++k )
				V[i][k+1][j] = U[i][k][j];
		}
	}

	for( int i = 0; i < num_map; ++i ){
		ret[i] = W[i]*V[i];
	}

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < ret[i].m; ++j )
			for( int k = 0; k < ret[i].n; ++k )
				ret[i][j][k] = (use_func ? activate_func(ret[i][j][k]) : ret[i][j][k]);
	
	return ret;
}

std::vector<std::vector<SparseFullyConnected::Vec>> SparseFullyConnected::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
{
	std::vector<Mat> tmp(num_map);
	for( int i = 0; i < num_map; ++i )
		tmp[i] = Mat(u[i][0].size(), u.size());

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < u.size(); ++j )
			for( int k = 0; k < u[i].size(); ++k )
				tmp[i][k][j] = u[j][i][k];
	
	auto U = apply(tmp);
	std::vector<std::vector<Vec>> ret(num_map);
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < U[i].n; ++j )
			for( int k = 0; k < U[i].m; ++k )
				ret[i][k][j] = U[j][i][k];

	return ret;
}

void SparseFullyConnected::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

	for( int i = 0; i < num_map; ++i ){
		ifs.read((char*)&W[i].m, sizeof(W[i].m));
		ifs.read((char*)&W[i].n, sizeof(W[i].n));
		for( int j = 0; j < W[i].m; ++j )
			for( int k = 0; k < W[i].n; ++k )
				ifs.read((char*)&W[i][j][k], sizeof(W[i][j][k]));
	}
}

void SparseFullyConnected::output_W ( const std::string& filename )
{
	std::ofstream ofs(filename, std::ios::binary);

	ofs.write((char*)&W[0].m, sizeof(W[0].m));
	ofs.write((char*)&W[0].n, sizeof(W[0].n));
	for( int i = 0; i < num_map; ++i ){
		ofs.write((char*)&W[i].m, sizeof(W[i].m));
		ofs.write((char*)&W[i].n, sizeof(W[i].n));
		for( int j = 0; j < W[i].m; ++j )
			for( int k = 0; k < W[i].n; ++k )
				ofs.write((char*)&W[i][j][k], sizeof(W[i][j][k]));
	}
}

#endif
