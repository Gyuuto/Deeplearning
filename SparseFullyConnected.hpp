#ifndef SPARSEFULLYCONNECTED_HPP
#define SPARSEFULLYCONNECTED_HPP

#include "Layer.hpp"

class SparseFullyConnected : public Layer
{
private:
	double RHO, BETA;
	std::vector<Mat> rho;
public:
	SparesFullyConnected ( int num_input, int num_output, double RHO, double BETA );

	std::vector<Mat> calc_delta ( const std::vector<Mat>& delta );
	std::vector<Mat> calc_gradient ( const std::vector<Mat>& delta );
	std::vector<Mat> apply ( const std::vector<Mat>& U );
	std::vector<Vec> apply ( const std::vector<Vec>& u );
};

SparseFullyConnected::SparseFullyConnected( int num_input, int num_output, double RHO, double BETA )
	: num_input(num_input), num_output(num_output), RHO(RHO), BETA(BETA)
{
	W.emplace_back(Mat(num_output, 1+num_input));
	rho.emplace_back(Mat(num_output, 1));
}

std::vector<Mat> SparseFullyConnected::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	// suppose a size of W and delta is 1.
	std::vector<Mat> tmp(1), nx_delta(1);
	tmp[0] = Mat::transpose(W[0])*delta[0];
	nx_delta[0] = Mat(tmp[0].m-1, tmp[0].n);

	const double R_LAMBDA = 0.9;
	for( int i = 0; i < U[0].m-1; ++i ){
		double tmp_rho = 0.0;
		for( int j = 0; j < U[0].n; ++j )
			tmp_rho += activate_func(U[0][i+1][j]);
		rho[i][0] = R_LAMBDA*rho[i][0] + (1.0-R_LAMBDA)*tmp_rho;
	}
	for( int i = 0; i < tmp[0].m; ++i ){
		double KL = (1.0-RHO)/(1.0-rho[i][0]) - RHO/rho[i][0];
		for( int j = 0; j < tmp[0].n; ++j )
			nx_delta[i][j] = (tmp[i+1][j]+BETA*KL)*activate_diff_func(U[0][i+1][j]);
	}
	
	return nx_delta;
}

std::vector<Mat> SparseFullyConnected::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	// suppose a size of W and delta is 1.
	std::vector<Mat> nabla(1);
	nabla[0] = Mat::zeros(W[i].m, W[i].n);
	for( int i = 0; i < nabla[0].m; ++i ){
		for( int j = 0; j < nabla[0].n; ++j ){
			for( int k = 0; k < delta[0].n; ++k ){
				nabla[0][i][j] += delta[0][i][k]*(
					j == 0 ? U[0][j][k] : activate_func(U[0][j][k])
					);
			}
		}
	}

	return nabla;
}

std::vector<Mat> SparseFullyConnected::apply ( const std::vector<Mat>& U )
{
	std::vector<Mat> ret(1);
	ret[0] = W*U;

	for( int i = 0; i < ret[0].m; ++i )
		for( int j = 0; j < ret[0].n; ++j )
			ret[0][i][j] = activate_func(ret[0][i][j]);
	
	return ret;
}

#endif
