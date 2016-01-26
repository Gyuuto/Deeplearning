#ifndef LAYER_HPP
#define LAYER_HPP

#include <string>
#include <vector>
#include <functional>

#include "matrix.hpp"

class Layer
{
protected:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	int num_map;
	std::vector<int> num_input, num_output;
	
	std::vector<Mat> W;
	std::function<double(double)> activate_func, activate_diff_func;
public:
	Layer(){}

	virtual std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta, const std::function<double(double)>& prev_activate_diff_func ) = 0;
	virtual std::vector<Mat> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta, const std::function<double(double)>& prev_activate_func ) = 0;
	virtual void update_W ( const std::vector<Mat>& dW ) = 0;

	virtual std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true ) = 0;
	virtual std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true ) = 0;

	
	std::vector<Mat> get_W ();
	std::tuple<std::function<double(double)>, std::function<double(double)>> get_function ();

	void set_W ( const std::vector<Mat>& W );
	void set_function ( const std::function<double(double)>& f,
						const std::function<double(double)>& d_f );
	
	virtual void output_W ( const std::string& filename ) = 0;
};

std::vector<Layer::Mat> Layer::get_W ()
{
	return this->W;
}

std::tuple<std::function<double(double)>, std::function<double(double)>> Layer::get_function ()
{
	return make_tuple(activate_func, activate_diff_func);
}

void Layer::set_W ( const std::vector<Mat>& W )
{
	this->W = W;
}

void Layer::set_function ( const std::function<double(double)>& f,
						   const std::function<double(double)>& d_f )
{
	activate_func = f;
	activate_diff_func = d_f;
}

#endif
