#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <memory>

#include "Layer.hpp"
#include "Matrix.hpp"

class Optimizer
{
protected:
	double learning_rate;
	std::shared_ptr<Layer> layer;

public:
	Optimizer(){}

	virtual void init ( Optimizer* opt, const std::shared_ptr<Layer>& layer ) = 0;
	virtual void update_W ( int iter, std::vector<std::vector<Matrix<double>>> nabla_w ) = 0;
};

#endif
