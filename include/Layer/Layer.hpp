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
#include "../mpi_helper.hpp"
#endif

#include "../Matrix.hpp"
#include "../Function.hpp"

template<template<typename> class Mat, typename Real>
class Layer
{
protected:
	bool is_use_bias;

	int prev_num_map, num_map;
	int prev_num_unit, num_unit;
	
#ifdef USE_MPI
	MPI_Comm inner_world, outer_world;
	int rank, nprocs;
#endif

	std::vector<std::vector<Mat<Real>>> W;
	std::shared_ptr<Function<Real>> func, prev_func;
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
	virtual std::vector<Mat<Real>> calc_delta ( const std::vector<Mat<Real>>& U, const std::vector<Mat<Real>>& delta ) = 0;
	virtual std::vector<std::vector<Mat<Real>>> calc_gradient ( const std::vector<Mat<Real>>& U, const std::vector<Mat<Real>>& delta ) = 0;
	virtual void update_W ( const std::vector<std::vector<Mat<Real>>>& dW ) = 0;

	virtual std::vector<Mat<Real>> apply ( const std::vector<Mat<Real>>& U, bool use_func = true ) = 0;

	virtual std::vector<std::vector<Mat<Real>>> get_W ();
	virtual std::shared_ptr<Function<Real>> get_function ();
	virtual std::shared_ptr<Function<Real>> get_prev_function ();

	virtual int get_num_map();
	virtual int get_num_unit();
	virtual int get_prev_num_map();
	virtual int get_prev_num_unit();
	
	virtual void set_W ( const std::vector<std::vector<Mat<Real>>>& W );
	virtual void set_function ( const std::shared_ptr<Function<Real>>& f );
	virtual void set_prev_function ( const std::shared_ptr<Function<Real>>& f );
	
	virtual void set_W ( const std::string& filename ) = 0;
	virtual void output_W ( const std::string& filename ) = 0;
	
#ifdef USE_MPI
	virtual void param_mix () = 0;
#endif
};

template<template<typename> class Mat, typename Real>
std::vector<std::vector<Mat<Real>>> Layer<Mat, Real>::get_W ()
{
	return this->W;
}

template<template<typename> class Mat, typename Real>
std::shared_ptr<Function<Real>> Layer<Mat, Real>::get_function ()
{
	return func;
}

template<template<typename> class Mat, typename Real>
std::shared_ptr<Function<Real>> Layer<Mat, Real>::get_prev_function ()
{
	return prev_func;
}

template<template<typename> class Mat, typename Real>
int Layer<Mat, Real>::get_num_map()
{
	return this->num_map;
}

template<template<typename> class Mat, typename Real>
int Layer<Mat, Real>::get_num_unit()
{
	return this->num_unit;
}

template<template<typename> class Mat, typename Real>
int Layer<Mat, Real>::get_prev_num_map()
{
	return this->prev_num_map;
}

template<template<typename> class Mat, typename Real>
int Layer<Mat, Real>::get_prev_num_unit()
{
	return this->prev_num_unit;
}

template<template<typename> class Mat, typename Real>
void Layer<Mat, Real>::set_W ( const std::vector<std::vector<Mat<Real>>>& W )
{
	this->W = W;
}

template<template<typename> class Mat, typename Real>
void Layer<Mat, Real>::set_function ( const std::shared_ptr<Function<Real>>& f )
{
	func = f;
}

template<template<typename> class Mat, typename Real>
void Layer<Mat, Real>::set_prev_function ( const std::shared_ptr<Function<Real>>& f )
{
	prev_func = f;
}

#ifdef USE_GPU
#include "Layer_gpu.hpp"
#endif

#endif
