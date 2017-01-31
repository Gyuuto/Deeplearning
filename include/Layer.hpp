#ifndef LAYER_HPP
#define LAYER_HPP

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "Matrix.hpp"
#include "Function.hpp"

class Layer
{
protected:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	bool is_use_bias;

	int prev_num_map, num_map;
	int prev_num_unit, num_unit;
	
#ifdef USE_MPI
	MPI_Comm inner_world, outer_world;
	int rank, nprocs;
#endif

	std::vector<std::vector<Mat>> W;
	std::shared_ptr<Function> func, prev_func;
public:
	double t_apply, t_delta, t_grad;
	double t_apply_init, t_apply_gemm, t_apply_repl, t_apply_comm;
	double t_delta_init, t_delta_gemm, t_delta_repl, t_delta_comm;
	double t_grad_init, t_grad_gemm, t_grad_repl, t_grad_comm;

	Layer(){}

#ifdef USE_MPI
	virtual void init ( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world ) = 0;
#else
	virtual void init ( std::mt19937& m ) = 0;
#endif
	virtual void finalize () = 0;
	virtual std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta ) = 0;
	virtual std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta ) = 0;
	virtual void update_W ( const std::vector<std::vector<Mat>>& dW ) = 0;

	virtual std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true ) = 0;
	virtual std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true ) = 0;

	// virtual std::map<std::string, double> get_error () = 0;
	
	virtual std::vector<std::vector<Mat>> get_W ();
	virtual std::shared_ptr<Function> get_function ();
	virtual std::shared_ptr<Function> get_prev_function ();

	virtual int get_num_map();
	virtual int get_num_unit();
	virtual int get_prev_num_map();
	virtual int get_prev_num_unit();
	
	virtual void set_W ( const std::vector<std::vector<Mat>>& W );
	virtual void set_function ( const std::shared_ptr<Function>& f );
	virtual void set_prev_function ( const std::shared_ptr<Function>& f );
	
	virtual void set_W ( const std::string& filename ) = 0;
	virtual void output_W ( const std::string& filename ) = 0;
	
#ifdef USE_MPI
	virtual void param_mix () = 0;
#endif
};

std::vector<std::vector<Layer::Mat>> Layer::get_W ()
{
	return this->W;
}

std::shared_ptr<Function> Layer::get_function ()
{
	return func;
}

std::shared_ptr<Function> Layer::get_prev_function ()
{
	return prev_func;
}

int Layer::get_num_map()
{
	return this->num_map;
}

int Layer::get_num_unit()
{
	return this->num_unit;
}

int Layer::get_prev_num_map()
{
	return this->prev_num_map;
}

int Layer::get_prev_num_unit()
{
	return this->prev_num_unit;
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
