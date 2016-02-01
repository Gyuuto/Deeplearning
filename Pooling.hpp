#ifndef POOLING_HPP
#define POOLING_HPP

#include "Layer.hpp"

#include <functional>

class Pooling : public Layer
{
private:
	int prev_ldu, ldu;
	int m, n, stlide;
public:
	Pooling( int prev_num_map, int prev_num_unit, int prev_ldu,
			 int num_map, int num_unit, int ldu,
			 int m, int n, int stlide, 
			 std::function<double(double)> activate_func, 
			 std::function<double(double)> activate_diff_func );

	void init ( std::mt19937& m );

	std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	void update_W ( const std::vector<std::vector<Mat>>& dW );

	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename );
};

Pooling::Pooling( int prev_num_map, int prev_num_unit, int prev_ldu,
				  int num_map, int num_unit, int ldu,
				  int m, int n, int stlide, 
				  std::function<double(double)> activate_func, 
				  std::function<double(double)> activate_diff_func )
{
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->prev_ldu = prev_ldu;

	this->num_map = num_map;
	this->num_unit = num_unit;
	this->ldu = ldu;

	this->m = m; this->n = n; this->stlide = stlide;

	this->activate_func = activate_func;
	this->activate_diff_func = activate_diff_func;
}

void Pooling::init ( std::mt19937& m )
{
	W = std::vector<std::vector<Mat>>();
}

std::vector<std::vector<Pooling::Mat>> Pooling::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	return std::vector<std::vector<Mat>>();
}

std::vector<Pooling::Mat> Pooling::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	std::vector<Mat> nx_delta(prev_num_map);

	for( int i = 0; i < prev_num_map; ++i ){
		nx_delta[i] = Mat(U[i].m, U[i].n);
		for( int j = 0; j < U[i].n; ++j )
			for( int x = 0; x < X; ++x )
				for( int y = 0; y < Y; ++y ){
					for( int s = -m/2; s < (m+1)/2; ++s )
						for( int t = -n/2; t < (n+1)/2; ++t ){
							int nx = x/stlide + s, ny = y/stlide + t;
							if( nx < 0 || nx >= X|| ny < 0 || ny >= Y ) continue;
							nx_delta[i][x+y*prev_ldu][j] += delta[i][nx+ny*ldu][j]*
								prev_activate_diff_func(U[i][x+y*prev_ldu][j]);
						}
				}
	}

	return nx_delta;
}

void Pooling::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	
}

std::vector<Pooling::Mat> Pooling::apply ( const std::vector<Mat>& U, bool use_func )
{
	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	std::vector<Mat> ret(num_map);

	for( int i = 0; i < num_map; ++i ){
		ret[i] = Mat(num_unit, U[0].n);
		for( int j = 0; j < prev_num_map; ++j ){
			for( int k = 0; k < U[0].n; ++k ){
				for( int y = 0; y < Y; y+=stlide )
					for( int x = 0; x < X; x+=stlide ){
						double val = U[j][0][k];

						for( int s = -m/2; s < (m+1)/2; ++s )
							for( int t = -n/2; t < (n+1)/2; ++t ){
								int nx = x+s, ny = y+t;
								if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;
								val = std::max(val, U[j][nx + ny*prev_ldu][k]);
							}

						ret[i][y/stlide*ldu + x/stlide][k] = val;
					}
			}
		}
	}
	
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < ret[i].m; ++j )
			for( int k = 0; k < ret[i].n; ++k )
				ret[i][j][k] = (use_func ? activate_func(ret[i][j][k]) : ret[i][j][k]);

	return ret;
}

std::vector<std::vector<Pooling::Vec>> Pooling::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
{
	std::vector<Mat> tmp(prev_num_map);
	for( int i = 0; i < prev_num_map; ++i )
		tmp[i] = Mat(u[i][0].size(), u.size());

	for( int i = 0; i < prev_num_map; ++i )
		for( int j = 0; j < u[i][0].size(); ++j )
			for( int k = 0; k < u.size(); ++k )
				tmp[i][j][k] = u[k][i][j];
	
	auto U = apply(tmp, use_func);
	std::vector<std::vector<Vec>> ret(U[0].n);
	for( int i = 0; i < U[0].n; ++i ){
		ret[i] = std::vector<Vec>(U.size(), Vec(U[0].m));
		for( int j = 0; j < U.size(); ++j )
			for( int k = 0; k < U[0].m; ++k )
				ret[i][j][k] = U[j][k][i];
	}

	return ret;
}

void Pooling::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);
}

void Pooling::output_W ( const std::string& filename )
{
	std::ofstream ofs(filename, std::ios::binary);
}

#endif
