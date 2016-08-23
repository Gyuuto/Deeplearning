#ifndef SGD_HPP
#define SGD_HPP

#include "Optimizer.hpp"

class SGD : public Optimizer
{
private:
	int update_iter;
	double dec_learning_rate;
public:
	SGD ( double learning_rate, int update_iter = -1, double dec_learning_rate = 0.0 );

	void init( const Optimizer* opt, const std::shread_ptr<Layer>& layer );
	void update_W ( int iter, std::vector<Matrix<T>> nabla_w );
};

SGD::SGD ( double learning_rate, int update_iter, double dec_learning_rate )
	: learning_rate(learning_rate), update_iter(update_iter), dec_learning_rate(dec_learning_rate)
{
	
}

void SGD::init ( const Optimizer* opt, const std::shread_ptr<Layer>& layer )
{
	this->layer = layer;

	this->learning_rate = opt->learning_rate;
	this->update_iter = opt->update_iter;
	this->dec_learning_rate = opt->dec_learning_rate;
}

void SGD::update_W ( int iter, std::vector<std::vector<Matrix<T>>> nabla_w )
{
	if( update_iter != -1 && iter % update_iter == 0 )
		learning_rate *= dec_learning_rate;

	for( int i = 0; i < nabla_w.size(); ++i )
		for( int j = 0; j < nabla_w[i].size(); ++j )
			nabla_w[i][j] *= -learning_rate;
	
	layer->update_W(nabla_w);
}

#endif
