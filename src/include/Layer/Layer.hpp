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
#include <mpi_helper.hpp>
#endif

#include <Matrix.hpp>
#ifdef USE_OPENCL
#include <clMatrix.hpp>
#endif
#ifdef USE_CUDA
#include <CUDA/cuda_kernel.h>
#include <cudaMatrix.hpp>
#endif
#include <Function.hpp>

template<template<typename> class Mat, typename Real>
class Layer
{
protected:
	std::string layer_name;
	bool is_use_bias, is_learning;
#ifdef USE_OPENCL
	cl_mem cl_use_bias;
#endif
	
	int prev_num_map, num_map;
	int prev_num_unit, num_unit;
	
#ifdef USE_MPI
	MPI_Comm inner_world, outer_world;
	int rank, nprocs;
#endif

	std::vector<Mat<Real>> W;
	std::vector<Mat<Real>> b;
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
	virtual void set_learning ();
	virtual void unset_learning ();
	virtual void finalize () = 0;


	virtual void calc_delta ( const Mat<Real>& U_apply, const Mat<Real>& U_diff, const Mat<Real>& delta, Mat<Real>& nx_delta ) = 0;
	virtual void calc_gradient ( const Mat<Real>& U_apply, const Mat<Real>& U_diff, const Mat<Real>& delta, std::vector<Mat<Real>>& nabla_W, std::vector<Mat<Real>>& nabla_b ) = 0;
	virtual void apply ( const Mat<Real>& U, Mat<Real>& ret, bool use_func = true ) = 0;
	virtual Mat<Real> apply ( const Mat<Real>& U, bool use_func = true ) = 0;

	virtual void update_W ( const std::vector<Mat<Real>>& dW, const std::vector<Mat<Real>>& db ) = 0;

	virtual const std::vector<Mat<Real>>& get_W ();
	virtual const std::vector<Mat<Real>>& get_b ();
	virtual std::shared_ptr<Function<Real>> get_function ();
	virtual std::shared_ptr<Function<Real>> get_prev_function ();

	virtual std::string get_layer_name();
	virtual bool get_learning ();
	virtual int get_num_map();
	virtual int get_num_unit();
	virtual int get_prev_num_map();
	virtual int get_prev_num_unit();
	
	virtual void set_W ( const std::vector<Mat<Real>>& W );
	virtual void set_b ( const std::vector<Mat<Real>>& b );
	virtual void set_function ( const std::shared_ptr<Function<Real>>& f );
	virtual void set_prev_function ( const std::shared_ptr<Function<Real>>& f );
	
	virtual void set_W ( const std::string& filename ) = 0;
	virtual void output_W ( const std::string& filename ) = 0;
	
#ifdef USE_MPI
	virtual void param_mix () = 0;
#endif
};

template<template<typename> class Mat, typename Real>
void Layer<Mat, Real>::set_learning ()
{
	is_learning = true;
}
template<template<typename> class Mat, typename Real>
void Layer<Mat, Real>::unset_learning ()
{
	is_learning = false;
}

template<template<typename> class Mat, typename Real>
const std::vector<Mat<Real>>& Layer<Mat, Real>::get_W ()
{
	return this->W;
}

template<template<typename> class Mat, typename Real>
const std::vector<Mat<Real>>& Layer<Mat, Real>::get_b ()
{
	return this->b;
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
std::string Layer<Mat, Real>::get_layer_name()
{
	return this->layer_name;
}

template<template<typename> class Mat, typename Real>
bool Layer<Mat, Real>::get_learning()
{
	return this->is_learning;
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
void Layer<Mat, Real>::set_W ( const std::vector<Mat<Real>>& W )
{
	this->W = W;
}

template<template<typename> class Mat, typename Real>
void Layer<Mat, Real>::set_b ( const std::vector<Mat<Real>>& b )
{
	this->b = b;
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
#endif
