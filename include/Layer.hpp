#ifndef LAYER_HPP
#define LAYER_HPP

#include <string>
#include <vector>
#include <functional>
#include <random>

#include "matrix.hpp"

class Layer
{
protected:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	int prev_num_map, num_map;
	int prev_num_unit, num_unit;
	
	std::vector<std::vector<Mat>> W;
	std::function<double(double)> activate_func, activate_diff_func;
	std::function<double(double)> prev_activate_func, prev_activate_diff_func;
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
	
	std::vector<std::vector<Mat>> get_W ();
	std::tuple<std::function<double(double)>, std::function<double(double)>> get_function ();

	void set_W ( const std::vector<std::vector<Mat>>& W );
	void set_function ( const std::function<double(double)>& f,
						const std::function<double(double)>& d_f );
	void set_prev_function ( const std::function<double(double)>& f,
							 const std::function<double(double)>& d_f );
	
	virtual void set_W ( const std::string& filename ) = 0;
	virtual void output_W ( const std::string& filename ) = 0;
};

std::vector<std::vector<Layer::Mat>> Layer::get_W ()
{
	return this->W;
}

std::tuple<std::function<double(double)>, std::function<double(double)>> Layer::get_function ()
{
	return make_tuple(activate_func, activate_diff_func);
}

void Layer::set_W ( const std::vector<std::vector<Mat>>& W )
{
	this->W = W;
}

void Layer::set_function ( const std::function<double(double)>& f,
						   const std::function<double(double)>& d_f )
{
	activate_func = f;
	activate_diff_func = d_f;
}

void Layer::set_prev_function ( const std::function<double(double)>& f,
								const std::function<double(double)>& d_f )
{
	prev_activate_func = f;
	prev_activate_diff_func = d_f;
}

#endif
