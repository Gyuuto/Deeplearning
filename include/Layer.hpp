#ifndef LAYER_HPP
#define LAYER_HPP

#include <string>
#include <vector>
#include <memory>
#include <random>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "matrix.hpp"
#include "Function.hpp"

class Layer
{
protected:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	int prev_num_map, num_map;
	int prev_num_unit, num_unit;
	
	std::vector<std::vector<Mat>> W;
	std::shared_ptr<Function> func, prev_func;
public:
	Layer(){}

	virtual void init ( std::mt19937& m ) = 0;
	virtual void finalize () = 0;
	virtual std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta ) = 0;
	virtual std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta ) = 0;
	virtual void update_W ( const std::vector<std::vector<Mat>>& dW ) = 0;

	virtual std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true ) = 0;
	virtual std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true ) = 0;

	// virtual std::map<std::string, double> get_error () = 0;
	
	virtual std::vector<std::vector<Mat>> get_W ();
	virtual std::shared_ptr<Function> get_function ();

	virtual void set_W ( const std::vector<std::vector<Mat>>& W );
	virtual void set_function ( const std::shared_ptr<Function>& f );
	virtual void set_prev_function ( const std::shared_ptr<Function>& f );
	
	virtual void set_W ( const std::string& filename ) = 0;
	virtual void output_W ( const std::string& filename ) = 0;
};

std::vector<std::vector<Layer::Mat>> Layer::get_W ()
{
	return this->W;
}

std::shared_ptr<Function> Layer::get_function ()
{
	return func;
}

void Layer::set_W ( const std::vector<std::vector<Mat>>& W )
{
	this->W = W;
}

void Layer::set_function ( const std::shared_ptr<Function>& f )
{
	func = f;
}

void Layer::set_prev_function ( const std::shared_ptr<Function>& f )
{
	prev_func = f;
}

#endif
