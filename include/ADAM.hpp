#ifndef ADAM_HPP
#define ADAM_HPP

#include "Function.hpp"

class ADAM : public Optimizer
{
private:
	double BETA, GAMMA, EPS; // parameter
	double beta_, gamma_; // temp variable for power value
	std::vector<std::vector<Matrix<double>>> v, r;
public:
	ADAM():beta_(1.0), gamma_(1.0){}
	ADAM( double learning_rate, double beta = 0.9, double gamma = 0.999, double eps = 1.0E-8 );

	void init ( Optimizer* opt, const std::shared_ptr<Layer>& layer );
	void update_W ( int iter, std::vector<std::vector<Matrix<double>>> nabla_w );
};

ADAM::ADAM( double learning_rate, double BETA, double GAMMA, double EPS )
	: BETA(BETA), GAMMA(GAMMA), EPS(EPS), beta_(1.0), gamma_(1.0)
{
	this->learning_rate = learning_rate;
}

void ADAM::init( Optimizer* opt, const std::shared_ptr<Layer>& layer )
{
	ADAM* opt_ = dynamic_cast<ADAM*>(opt);
	this->layer = layer;

	this->learning_rate = opt_->learning_rate;
	this->BETA = opt_->BETA; this->GAMMA = opt_->GAMMA; this->EPS = opt_->EPS;
	
	auto W = layer->get_W();
	int num_map = W.size(), prev_num_map = W[0].size();

	v = std::vector<std::vector<Matrix<double>>>(num_map);
	r = std::vector<std::vector<Matrix<double>>>(num_map);
	for( int i = 0; i < num_map; ++i ){
		v[i] = std::vector<Matrix<double>>(prev_num_map);
		r[i] = std::vector<Matrix<double>>(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j ){
			v[i][j] = Matrix<double>::zeros(W[i][j].m, W[i][j].n);
			r[i][j] = Matrix<double>::zeros(W[i][j].m, W[i][j].n);
		}
	}
}

void ADAM::update_W ( int iter, std::vector<std::vector<Matrix<double>>> nabla_w )
{
	for( int i = 0; i < nabla_w.size(); ++i )
		for( int j = 0; j < nabla_w[i].size(); ++j ){
			v[i][j] = BETA*v[i][j] + (1.0 - BETA)*nabla_w[i][j];
			r[i][j] = GAMMA*r[i][j] + (1.0 - GAMMA)*Matrix<double>::hadamard(nabla_w[i][j], nabla_w[i][j]);
		}

	beta_ *= BETA; gamma_ *= GAMMA;
	for( int i = 0; i < nabla_w.size(); ++i )
		for( int j = 0; j < nabla_w[i].size(); ++j ){
			auto v_hat = v[i][j]/(1.0 - beta_);
			auto r_hat = r[i][j]/(1.0 - gamma_);
			nabla_w[i][j] = -learning_rate * v_hat / (Sqrt()(r_hat, false) + EPS);
		}

	layer->update_W(nabla_w);
}

#endif
