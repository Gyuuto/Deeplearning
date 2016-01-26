#ifndef FULLYCONNECTED_HPP
#define FULLYCONNECTED_HPP

#include <fstream>

#include <chrono>
#include <random>

#include "Layer.hpp"

class FullyConnected : public Layer
{
private:

public:
	FullyConnected ( int num_map, int num_input, int num_output, 
					 const std::function<double(double)>& f,
					 const std::function<double(double)>& d_f );

	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta, const std::function<double(double)>& prev_activate_diff_func );
	std::vector<Mat> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta, const std::function<double(double)>& prev_activate_func );
	void update_W ( const std::vector<Mat>& dW );

	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );

	void output_W ( const std::string& filename );
};

FullyConnected::FullyConnected( int num_map, int num_input, int num_output,
								const std::function<double(double)>& f,
								const std::function<double(double)>& d_f )
{
	this->num_map = num_map;
	this->num_input.emplace_back(num_input);
	this->num_output.emplace_back(num_output);

	const double r = sqrt(6.0/(num_input + num_output));

	std::mt19937 m(100);
	std::uniform_real_distribution<double> d_rand(-r, r);
	
	for( int i = 0; i < num_map; ++i ){
		W.emplace_back(num_output, 1+num_input);
		for( int j = 0; j < W[i].m; ++j )
			for( int k = 0; k < W[i].n; ++k )
				W[i][j][k] = d_rand(m);
	}

	activate_func = f;
	activate_diff_func = d_f;
}

std::vector<FullyConnected::Mat> FullyConnected::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta, const std::function<double(double)>& prev_activate_diff_func )
{
	std::vector<Mat> tmp(num_map), nx_delta(num_map);
	for( int i = 0; i < num_map; ++i ){
		tmp[i] = Mat::transpose(W[i])*delta[i];
		nx_delta[i] = Mat(tmp[i].m-1, tmp[i].n);
	}

	for( int i = 0; i < num_map; ++i ) 
		for( int j = 0; j < tmp[i].m-1; ++j )
			for( int k = 0; k < tmp[i].n; ++k )
				nx_delta[i][j][k] = tmp[i][j+1][k]*prev_activate_diff_func(U[i][j][k]);
	
	return nx_delta;
}

std::vector<FullyConnected::Mat> FullyConnected::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta, const std::function<double(double)>& prev_activate_func )
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

void FullyConnected::update_W ( const std::vector<Mat>& dW )
{
	for( int i = 0; i < W.size(); ++i )
		W[i] = W[i] + dW[i];
}

std::vector<FullyConnected::Mat> FullyConnected::apply ( const std::vector<Mat>& U, bool use_func )
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
		// printf("(%d %d) (%d %d)\n", W[i].m, W[i].n, V[i].m, V[i].n);
		ret[i] = W[i]*V[i];
	}

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < ret[i].m; ++j )
			for( int k = 0; k < ret[i].n; ++k )
				ret[i][j][k] = (use_func ? activate_func(ret[i][j][k]) : ret[i][j][k]);
	
	return ret;
}

std::vector<std::vector<FullyConnected::Vec>> FullyConnected::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
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

void FullyConnected::output_W ( const std::string& filename )
{
	std::ofstream ofs(filename, std::ios::binary);

	ofs.write((char*)&W[0].m, sizeof(W[0].m));
	ofs.write((char*)&W[0].n, sizeof(W[0].n));
	for( int i = 0; i < W[0].m; ++i )
		for( int j = 0; j < W[0].n; ++j )
			ofs.write((char*)&W[0][i][j], sizeof(W[0][i][j]));
}

#endif
