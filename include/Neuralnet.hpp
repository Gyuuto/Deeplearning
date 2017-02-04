#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <random>

#ifdef DEBUG
#include <chrono>
#endif

#include "Layer/Layer.hpp"
#include "Function.hpp"
#include "Matrix.hpp"
#ifdef USE_GPU
#include "clMatrix.hpp"
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

template<template<typename> class Mat, typename Real>
class Neuralnet
{
private:
	Real adam_beta = 0.9, adam_gamma = 0.999, adam_eps = 1.0E-8;
	std::vector<std::vector<std::vector<Mat<Real>>>> adam_v_W, adam_r_W, adam_v_b, adam_r_b;
	Real adam_beta_ = 1.0, adam_gamma_ = 1.0;

	int BATCH_SIZE, UPDATE_ITER;
	Real EPS, LAMBDA;

	std::shared_ptr<LossFunction<Real>> loss;
	std::vector<std::shared_ptr<Layer<Mat, Real>>> layer;
	
	std::mt19937 mt;
	std::uniform_real_distribution<Real> d_rand;

#ifdef USE_MPI
	MPI_Comm outer_world, inner_world;
#endif

	std::pair<std::vector<std::vector<std::vector<Mat<Real>>>>, std::vector<std::vector<std::vector<Mat<Real>>>>> calc_gradient (const std::vector<std::vector<Mat<Real>>>& U, const std::vector<Mat<Real>>& d);
	void check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<Matrix<Real>>& X, const std::vector<Matrix<Real>>& Y, const std::vector<std::vector<std::vector<Matrix<Real>>>>& nabla_w, const std::vector<std::vector<std::vector<Matrix<Real>>>>& nabla_b );
#ifdef USE_GPU
	void check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<clMatrix<Real>>& X, const std::vector<clMatrix<Real>>& Y, const std::vector<std::vector<std::vector<clMatrix<Real>>>>& nabla_w, const std::vector<std::vector<std::vector<clMatrix<Real>>>>& nabla_b, cl_mem& cl_N, cl_mem& cl_idx, cl_mem& cl_offset );
#endif
public:
#ifdef USE_MPI
	Neuralnet( const std::shared_ptr<LossFunction<Real>>& loss, MPI_Comm outer_world, MPI_Comm inner_world );
#else
	Neuralnet( const std::shared_ptr<LossFunction<Real>>& loss );
#endif

	void set_EPS ( const Real& EPS );
	void set_LAMBDA ( const Real& LAMBDA );
	void set_BATCHSIZE ( const int& BATCH_SIZE );
	void set_UPDATEITER ( const int& UPDATE_ITER );

	void add_layer( const std::shared_ptr<Layer<Mat, Real>>& layer );

#ifdef USE_MPI
	void averaging ();
#endif
	void learning ( const std::vector<Matrix<Real>>& X, const std::vector<Matrix<Real>>& Y,
					const int MAX_ITER = 1000,
					const std::function<void(Neuralnet<Matrix, Real>&, const int, const std::vector<Matrix<Real>>&, const std::vector<Matrix<Real>>&)>& each_func
					= [](Neuralnet<Matrix, Real>& nn, const int epoch, const std::vector<Matrix<Real>>& x, const std::vector<Matrix<Real>>& d) -> void {} );
#ifdef USE_GPU
	void learning ( const std::vector<clMatrix<Real>>& X, const std::vector<clMatrix<Real>>& Y,
					const int MAX_ITER = 1000,
					const std::function<void(Neuralnet<clMatrix, Real>&, const int, const std::vector<clMatrix<Real>>&, const std::vector<clMatrix<Real>>&)>& each_func
					= [](Neuralnet<clMatrix, Real>& nn, const int epoch, const std::vector<clMatrix<Real>>& x, const std::vector<clMatrix<Real>>& d) -> void {} );	
#endif

	std::vector<Mat<Real>> apply ( const std::vector<Mat<Real>>& X ) const;

	void print_cost ( const std::vector<Mat<Real>>& x, const std::vector<Mat<Real>>& y ) const;

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename ) const;
};

//////////////////// PRIVATE FUNCTION ////////////////////
template<template<typename> class Mat, typename Real>
std::pair<std::vector<std::vector<std::vector<Mat<Real>>>>, std::vector<std::vector<std::vector<Mat<Real>>>>> Neuralnet<Mat, Real>::calc_gradient (const std::vector<std::vector<Mat<Real>>>& U, const std::vector<Mat<Real>>& d)
{
	const int num_layer = layer.size();
	
	std::vector<Mat<Real>> delta(d.size());
	for( int i = 0; i < d.size(); ++i ) delta[i] = Mat<Real>(d[i].m, d[i].n);

	std::shared_ptr<Function<Real>> f = layer[num_layer-1]->get_function();
	for( int i = 0; i < d.size(); ++i )
		delta[i] = Mat<Real>::hadamard((*loss)((*f)(U[num_layer][i], false), d[i], true), (*f)(U[num_layer][i], true));

#ifdef DEBUG
	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(this->inner_world, &rank);
#endif
#endif
	std::vector<std::vector<std::vector<Mat<Real>>>> nabla_W(num_layer), nabla_b(num_layer);
	for( int i = num_layer-1; i >= 0; --i ){
#ifdef DEBUG
		auto beg1 = std::chrono::system_clock::now();
#endif
		std::tie(nabla_W[i], nabla_b[i]) = layer[i]->calc_gradient(U[i], delta);
#ifdef DEBUG
		auto end1 = std::chrono::system_clock::now();
#endif

		if( i == 0 ){
#ifdef DEBUG
			if( rank == 0 ) printf("  layer %d, calc grad : %3lld\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(end1 - beg1).count());
#endif
			continue;
		}
#ifdef DEBUG
		auto beg2 = std::chrono::system_clock::now();
#endif
		delta = layer[i]->calc_delta(U[i], delta);
#ifdef DEBUG
		auto end2 = std::chrono::system_clock::now();
		if( rank == 0 ) printf("  layer %d, calc grad : %3lld, calc delta %3lld\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(end1 - beg1).count(), std::chrono::duration_cast<std::chrono::milliseconds>(end2 - beg2).count());
#endif
	}

	return std::make_pair(nabla_W, nabla_b);
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<Matrix<Real>>& X, const std::vector<Matrix<Real>>& Y, const std::vector<std::vector<std::vector<Matrix<Real>>>>& nabla_w, const std::vector<std::vector<std::vector<Matrix<Real>>>>& nabla_b )
{
	int rank = 0, target_rank = 0;
	int num_layer = this->layer.size();

#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif
	
	std::vector<Matrix<Real>> tmp_X(X.size(), Matrix<Real>(X[0].m, BATCH_SIZE)),
		tmp_Y(Y.size(), Matrix<Real>(Y[0].m, BATCH_SIZE));
	for( int j = 0; j < X.size(); ++j ){
		auto tmp_input = X[j];
		for( int k = 0; k < X[0].m; ++k )
			for( int l = 0; l < BATCH_SIZE; ++l )
				tmp_X[j](k, l) = X[j](k, idx[(cnt+l)%X[j].n]);
	}
	for( int j = 0; j < Y.size(); ++j ){
		for( int k = 0; k < Y[0].m; ++k )
			for( int l = 0; l < BATCH_SIZE; ++l )
				tmp_Y[j](k, l) = Y[j](k, idx[(cnt+l)%Y[j].n]);
	}

	const double delta_x = 1.0E-3;

	// Calculate gradient numerically for confirmation of computing
	for( int i = 0; i < num_layer; ++i ){
		if( rank == target_rank ) printf("\tlayer %d\n", i);
		auto W = layer[i]->get_W();
		auto b = layer[i]->get_b();
		if( W.size() == 0 ) continue;

		int J = W.size(), K = W[0].size(), L = W[0][0].m, M = W[0][0].n;
#ifdef USE_MPI
		MPI_Bcast(&J, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&K, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&L, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&M, 1, MPI_INT, target_rank, inner_world);
#endif
		if( rank == target_rank ) printf("\t  grad W\n");
		for( int j = 0; j < std::min(2, J); ++j ){ // num_map
			for( int k = 0; k < std::min(2, K); ++k ){ // prev_num_map
				for( int l = 0; l < std::min(4, L); ++l ){
					for( int m = 0; m < std::min(4, M); ++m ){
						double tmp;
						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							tmp = W[j][k](l,m);
							tmp = delta_x*(std::abs(tmp) < 1.0E0 ? 1.0 : std::abs(tmp));
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							W[j][k](l,m) += tmp;
							layer[i]->set_W(W);
						}
						double E1 = 0.0;
						auto tmp1 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E1 += (*loss)(tmp1[n], tmp_Y[n], false)(0, 0);
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							W[j][k](l,m) -= 2*tmp;
							layer[i]->set_W(W);
						}
						double E2 = 0.0;
						auto tmp2 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E2 += (*loss)(tmp2[n], tmp_Y[n], false)(0, 0);
						}
						
						if( rank == target_rank ){
							double grad = nabla_w[i][j][k](l,m);
							printf("\t%3d, %3d, %3d, %3d : ( %.10E, %.10E = %.10E )\n",
								   j, k, l, m,
								   0.5*(E1 - E2)/tmp/BATCH_SIZE, grad, (std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE - grad))/std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE));
						}
					}
				}
			}
		}
		if( rank == target_rank ) printf("\t  grad b\n");
		for( int j = 0; j < std::min(2, J); ++j ){ // num_map
			for( int k = 0; k < std::min(2, K); ++k ){ // prev_num_map
				for( int l = 0; l < std::min(4, L); ++l ){
					for( int m = 0; m < std::min(1, M); ++m ){
						double tmp;
						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							tmp = b[j][k](l,m);
							tmp = delta_x*(std::abs(tmp) < 1.0E0 ? 1.0 : std::abs(tmp));
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							b[j][k](l,m) += tmp;
							layer[i]->set_b(b);
						}
						double E1 = 0.0;
						auto tmp1 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E1 += (*loss)(tmp1[n], tmp_Y[n], false)(0, 0);
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							b[j][k](l,m) -= 2*tmp;
							layer[i]->set_b(b);
						}
						double E2 = 0.0;
						auto tmp2 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E2 += (*loss)(tmp2[n], tmp_Y[n], false)(0, 0);
						}
						
						if( rank == target_rank ){
							double grad = nabla_b[i][j][k](l,m);
							printf("\t%3d, %3d, %3d, %3d : ( %.10E, %.10E = %.10E )\n",
								   j, k, l, m,
								   0.5*(E1 - E2)/tmp/BATCH_SIZE, grad, (std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE - grad))/std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE));
						}
					}
				}
			}
		}
		if( rank == target_rank ) puts("");
	}
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<clMatrix<Real>>& X, const std::vector<clMatrix<Real>>& Y, const std::vector<std::vector<std::vector<clMatrix<Real>>>>& nabla_w, const std::vector<std::vector<std::vector<clMatrix<Real>>>>& nabla_b, cl_mem& cl_N, cl_mem& cl_idx, cl_mem& cl_offset )
{
	int rank = 0, target_rank = 0;
	int num_layer = this->layer.size();

#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif
	
	std::vector<clMatrix<Real>> tmp_X(X.size(), clMatrix<Real>(X[0].m, BATCH_SIZE)),
		tmp_Y(Y.size(), clMatrix<Real>(Y[0].m, BATCH_SIZE));
	for( int i = 0; i < X.size(); ++i ){
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 0, &tmp_X[i].v );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 1, &X[i].v );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 2, &cl_idx );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 3, &cl_offset );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 4, &cl_N );
		cl_device_manager.run_kernel( PRG::ASSIGN_DATA, tmp_X[i].m, tmp_X[i].n );
	}
	for( int i = 0; i < Y.size(); ++i ){
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 0, &tmp_Y[i].v );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 1, &Y[i].v );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 2, &cl_idx );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 3, &cl_offset );
		cl_device_manager.set_argument( PRG::ASSIGN_DATA, 4, &cl_N );
		cl_device_manager.run_kernel( PRG::ASSIGN_DATA, tmp_Y[i].m, tmp_Y[i].n );
	}

	const double delta_x = 1.0E-4;
	
	// Calculate gradient numerically for confirmation of computing
	for( int i = 0; i < num_layer; ++i ){
		if( rank == target_rank ) printf("\tlayer %d\n", i);
		auto W = layer[i]->get_W();
		auto b = layer[i]->get_b();
		if( W.size() == 0 ) continue;

		int J = W.size(), K = W[0].size(), L = W[0][0].m, M = W[0][0].n;
#ifdef USE_MPI
		MPI_Bcast(&J, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&K, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&L, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&M, 1, MPI_INT, target_rank, inner_world);
#endif
		if( rank == target_rank ) printf("\t  grad W\n");
		for( int j = 0; j < std::min(2, J); ++j ){ // num_map
			for( int k = 0; k < std::min(2, K); ++k ){ // prev_num_map
				for( int l = 0; l < std::min(4, L); ++l ){
					for( int m = 0; m < std::min(4, M); ++m ){
						double tmp;
						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							tmp = W[j][k].get_element(l,m);
							tmp = delta_x*(std::abs(tmp) < 1.0E0 ? 1.0 : std::abs(tmp));
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							W[j][k].set_element(l,m, W[j][k].get_element(l,m)+tmp);
							layer[i]->set_W(W);
						}
						double E1 = 0.0;
						auto tmp1 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E1 += (*loss)(tmp1[n], tmp_Y[n], false).get_element(0, 0);
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							W[j][k].set_element(l,m, W[j][k].get_element(l,m) - 2*tmp);
							layer[i]->set_W(W);
						}
						double E2 = 0.0;
						auto tmp2 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E2 += (*loss)(tmp2[n], tmp_Y[n], false).get_element(0, 0);
						}

						if( rank == target_rank ){
							double grad = nabla_w[i][j][k].get_element(l,m);
							printf("\t%3d, %3d, %3d, %3d : ( %.10E, %.10E = %.10E )\n",
								   j, k, l, m,
								   0.5*(E1 - E2)/tmp/BATCH_SIZE, grad, (std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE - grad))/std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE));
						}
					}
				}
			}
		}
		if( rank == target_rank ) printf("\t  grad b\n");
		for( int j = 0; j < std::min(2, J); ++j ){ // num_map
			for( int k = 0; k < std::min(2, K); ++k ){ // prev_num_map
				for( int l = 0; l < std::min(4, L); ++l ){
					for( int m = 0; m < std::min(1, M); ++m ){
						double tmp;
						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							tmp = b[j][k].get_element(l,m);
							tmp = delta_x*(std::abs(tmp) < 1.0E0 ? 1.0 : std::abs(tmp));
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							b[j][k].set_element(l,m, b[j][k].get_element(l,m)+tmp);
							layer[i]->set_b(b);
						}
						double E1 = 0.0;
						auto tmp1 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E1 += (*loss)(tmp1[n], tmp_Y[n], false).get_element(0, 0);
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
							b[j][k].set_element(l,m, b[j][k].get_element(l,m) - 2*tmp);
							layer[i]->set_b(b);
						}
						double E2 = 0.0;
						auto tmp2 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
							E2 += (*loss)(tmp2[n], tmp_Y[n], false).get_element(0, 0);
						}
						
						if( rank == target_rank ){
							double grad = nabla_b[i][j][k].get_element(l,m);
							printf("\t%3d, %3d, %3d, %3d : ( %.10E, %.10E = %.10E )\n",
								   j, k, l, m,
								   0.5*(E1 - E2)/tmp/BATCH_SIZE, grad, (std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE - grad))/std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE));
						}
					}
				}
			}
		}
		if( rank == target_rank ) puts("");
	}
}
#endif

//////////////////// PUBLIC FUNCTION ////////////////////
template<template<typename> class Mat, typename Real>
#ifdef USE_MPI
Neuralnet<Mat, Real>::Neuralnet( const std::shared_ptr<LossFunction<Real>>& loss, MPI_Comm outer_world, MPI_Comm inner_world )
	:EPS(1.0E-3), LAMBDA(0.0), BATCH_SIZE(1), UPDATE_ITER(-1), loss(loss), outer_world(outer_world), inner_world(inner_world)
#else
Neuralnet<Mat, Real>::Neuralnet( const std::shared_ptr<LossFunction<Real>>& loss )
	:EPS(1.0E-3), LAMBDA(0.0), BATCH_SIZE(1), UPDATE_ITER(-1), loss(loss)
#endif	 
{
	int rank = 0, seed;
#ifdef USE_MPI
	MPI_Comm_rank(outer_world, &rank);
#endif
	
	seed = time(NULL) + rank;
#ifdef USE_MPI
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, inner_world);
#endif
	mt = std::mt19937(seed);
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::set_EPS ( const Real& EPS )
{
	this->EPS = EPS;
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::set_LAMBDA ( const Real& LAMBDA )
{
	this->LAMBDA = LAMBDA;
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::set_BATCHSIZE ( const int& BATCH_SIZE )
{
	this->BATCH_SIZE = BATCH_SIZE;
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::set_UPDATEITER ( const int& UPDATE_ITER )
{
	this->UPDATE_ITER = UPDATE_ITER;
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::add_layer( const std::shared_ptr<Layer<Mat, Real>>& layer )
{
	std::shared_ptr<Function<Real>> f;
	int prev_num_unit = -1, prev_num_map = -1;

	if( this->layer.size() == 0 )
		f = std::shared_ptr<Function<Real>>(new Identity<Real>);
	else{
		prev_num_unit = this->layer[this->layer.size()-1]->get_num_unit();
		prev_num_map = this->layer[this->layer.size()-1]->get_num_map();
		f = this->layer[this->layer.size()-1]->get_function();
	}

	if( prev_num_unit != -1 &&
		(layer->get_prev_num_map() != prev_num_map || layer->get_prev_num_unit() != prev_num_unit)
		){
		int rank = 0;
#ifdef USE_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		if( rank == 0 ){
			if( layer->get_prev_num_map() != prev_num_map )
				printf("WARNING : Wrong prev_num_map on layer %lu.\n  Estimate prev_num_map = %d.\n",
					   this->layer.size() + 1, prev_num_map);
			if( layer->get_prev_num_unit() != prev_num_unit )
				printf("WARNING : Wrong prev_num_unit on layer %lu.\n  Estimate prev_num_unit = %d.\n",
					   this->layer.size() + 1, prev_num_unit);
		}
	}
	
	this->layer.emplace_back( layer );

	int idx = this->layer.size()-1;
	this->layer[idx]->set_prev_function(f);
#ifdef USE_MPI
	this->layer[idx]->init(mt, inner_world, outer_world);
#else
	this->layer[idx]->init(mt);
#endif

	auto w = layer->get_W();
	adam_v_W.push_back(std::vector<std::vector<Mat<Real>>>(w.size()));
	adam_r_W.push_back(std::vector<std::vector<Mat<Real>>>(w.size()));
	for( int j = 0; j < w.size(); ++j ){
		for( int k = 0; k < w[j].size(); ++k ){
			adam_v_W[idx][j].push_back(Mat<Real>::zeros(w[j][k].m, w[j][k].n));
			adam_r_W[idx][j].push_back(Mat<Real>::zeros(w[j][k].m, w[j][k].n));
		}
	}

	auto b = layer->get_b();
	adam_v_b.push_back(std::vector<std::vector<Mat<Real>>>(b.size()));
	adam_r_b.push_back(std::vector<std::vector<Mat<Real>>>(b.size()));
	for( int j = 0; j < b.size(); ++j ){
		for( int k = 0; k < b[j].size(); ++k ){
			adam_v_b[idx][j].push_back(Mat<Real>::zeros(b[j][k].m, b[j][k].n));
			adam_r_b[idx][j].push_back(Mat<Real>::zeros(b[j][k].m, b[j][k].n));
		}
	}
}

#ifdef USE_MPI
template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::averaging ()
{
	for( int i = 0; i < layer.size(); ++i ){
		layer[i]->param_mix();
	}
}
#endif

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::learning ( const std::vector<Matrix<Real>>& X, const std::vector<Matrix<Real>>& Y,
									  const int MAX_ITER,
									  const std::function<void(Neuralnet<Matrix, Real>&, const int, const std::vector<Matrix<Real>>&, const std::vector<Matrix<Real>>&)>& each_func )
{
	int nprocs = 1, myrank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &myrank);
	MPI_Comm_size(inner_world, &nprocs);
#endif

	const int num_layer = layer.size();
	const int num_data = X[0].n;

	int seed = time(NULL);
#ifdef USE_MPI
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, inner_world);
#endif
	mt = std::mt19937(seed);

	// index of data
	std::vector<int> idx(num_data);
	std::iota(idx.begin(), idx.end(), 0);
	std::shuffle( idx.begin(), idx.end(), mt );

	// memory allocation for matrix U and D.
	std::vector<Mat<Real>> D(Y.size(), Mat<Real>(Y[0].m, BATCH_SIZE));
	std::vector<std::vector<Mat<Real>>> U(num_layer+1);
	U[0] = std::vector<Mat<Real>>(layer[0]->get_prev_num_map(), Mat<Real>(layer[0]->get_prev_num_unit(), BATCH_SIZE));
	for( int i = 0; i < U.size()-1; ++i ){
		U[i+1] = std::vector<Mat<Real>>(layer[i]->get_num_map(), Mat<Real>(layer[i]->get_num_unit(), BATCH_SIZE));
	}

	each_func(*this, 0, U[0], D);

	int cnt = 0;
	for( int n = 0; n < MAX_ITER; ++n ){
#ifdef DEBUG
		auto beg = std::chrono::system_clock::now();
#endif
		// assign data to mini-batch
#pragma omp parallel
		{
			for( int i = 0; i < X.size(); ++i )
#pragma omp for schedule(auto) nowait
				for( int j = 0; j < U[0][i].m; ++j )
					for( int k = 0; k < BATCH_SIZE; ++k )
						U[0][i](j,k) = X[i](j, idx[(cnt+k)%num_data]);
		
			for( int i = 0; i < Y.size(); ++i )
#pragma omp for schedule(auto) nowait
				for( int j = 0; j < D[i].m; ++j )
					for( int k = 0; k < BATCH_SIZE; ++k )
						D[i](j,k) = Y[i](j, idx[(cnt+k)%num_data]);
		}
#ifdef DEBUG
		auto end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Init : %3lld %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count(), n);
#endif
#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
		// feed forward calculation
		for( int i = 0; i < num_layer; ++i ) {
#ifdef DEBUG
			auto beg = std::chrono::system_clock::now();
#endif
			auto V = U[i];
			if( i != 0 ){
				std::shared_ptr<Function<Real>> f = layer[i-1]->get_function();
				
				for( int j = 0; j < V.size(); ++j )
					V[j] = (*f)(V[j], false);
			}

			auto tmp = layer[i]->apply(V, false);
			for( int j = 0; j < tmp.size(); ++j ){
				U[i+1][j] = tmp[j];
			}
#ifdef DEBUG
			auto end = std::chrono::system_clock::now();
			if( myrank == 0 ) printf("  layer %d : %3lld\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif
		}
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Feed : %3lld %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count(), n);
#endif

		// back propagation calculation
#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
		std::vector<std::vector<std::vector<Matrix<Real>>>> nabla_W, nabla_b;
		tie(nabla_W, nabla_b) = calc_gradient(U, D);
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Back : %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif

#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
		// averaging all gradients of weights of mini-batches
		for( int i = 0; i < nabla_W.size(); ++i )
			for( int j = 0; j < nabla_W[i].size(); ++j )
				for( int k = 0; k < nabla_W[i][j].size(); ++k )
					nabla_W[i][j][k] *= 1.0/BATCH_SIZE;	
		for( int i = 0; i < nabla_b.size(); ++i )
			for( int j = 0; j < nabla_b[i].size(); ++j )
				for( int k = 0; k < nabla_b[i][j].size(); ++k )
					nabla_b[i][j][k] *= 1.0/BATCH_SIZE;
	
#ifdef CHECK_GRAD
		check_gradient(cnt, idx, X, Y, nabla_W, nabla_b);
#endif
		cnt += BATCH_SIZE;
		if( cnt >= num_data ){
			std::shuffle( idx.begin(), idx.end(), mt );
			cnt = 0;
		}

		// update W
		adam_beta_ *= adam_beta;
		adam_gamma_ *= adam_gamma;
		for( int i = 0; i < num_layer; ++i ){
			auto W = layer[i]->get_W();
			auto b = layer[i]->get_b();

			if( W.size() == 0 ) continue;

			std::vector<std::vector<Mat<Real>>> update_W(W.size(), std::vector<Mat<Real>>(W[0].size(), Mat<Real>(W[0][0].m, W[0][0].n))), update_b(b.size(), std::vector<Mat<Real>>(b[0].size(), Mat<Real>(b[0][0].m, b[0][0].n)));
			if( std::abs(LAMBDA) > 1.0E-15 ){
				// L2 norm regularization
				for( int j = 0; j < W.size(); ++j )
					for( int k = 0; k < W[j].size(); ++k )
						nabla_W[i][j][k] += LAMBDA*W[j][k];
			}

#pragma omp parallel
			{
				// ADAM
				for( int j = 0; j < nabla_W[i].size(); ++j )
					for( int k = 0; k < nabla_W[i][j].size(); ++k )
#pragma omp for schedule(auto) nowait
						for( int l = 0; l < nabla_W[i][j][k].m; ++l )
							for( int m = 0; m < nabla_W[i][j][k].n; ++m ){
								adam_v_W[i][j][k](l,m) = adam_beta*adam_v_W[i][j][k](l,m) + (1.0 - adam_beta)*nabla_W[i][j][k](l,m);
								adam_r_W[i][j][k](l,m) = adam_gamma*adam_r_W[i][j][k](l,m) + (1.0 - adam_gamma)*(nabla_W[i][j][k](l,m)*nabla_W[i][j][k](l,m));
							}

#pragma omp barrier
				
				for( int j = 0; j < W.size(); ++j )
					for( int k = 0; k < W[j].size(); ++k ){
#pragma omp for schedule(auto) nowait
						for( int l = 0; l < update_W[j][k].m; ++l )
							for( int m = 0; m < update_W[j][k].n; ++m ){
								auto v_hat = adam_v_W[i][j][k](l,m) / (1.0 - adam_beta_);
								auto r_hat = adam_r_W[i][j][k](l,m) / (1.0 - adam_gamma_);
								update_W[j][k](l,m) = -EPS*v_hat/(sqrt(r_hat)+adam_eps);
							}
					}
				
				for( int j = 0; j < nabla_b[i].size(); ++j )
					for( int k = 0; k < nabla_b[i][j].size(); ++k )
#pragma omp for schedule(auto) nowait
						for( int l = 0; l < nabla_b[i][j][k].m; ++l )
							for( int m = 0; m < nabla_b[i][j][k].n; ++m ){
								adam_v_b[i][j][k](l,m) = adam_beta*adam_v_b[i][j][k](l,m) + (1.0 - adam_beta)*nabla_b[i][j][k](l,m);
								adam_r_b[i][j][k](l,m) = adam_gamma*adam_r_b[i][j][k](l,m) + (1.0 - adam_gamma)*(nabla_b[i][j][k](l,m)*nabla_b[i][j][k](l,m));
							}

#pragma omp barrier
				
				for( int j = 0; j < W.size(); ++j )
					for( int k = 0; k < W[j].size(); ++k ){
#pragma omp for schedule(auto) nowait
						for( int l = 0; l < update_b[j][k].m; ++l )
							for( int m = 0; m < update_b[j][k].n; ++m ){
								auto v_hat = adam_v_b[i][j][k](l,m) / (1.0 - adam_beta_);
								auto r_hat = adam_r_b[i][j][k](l,m) / (1.0 - adam_gamma_);
								update_b[j][k](l,m) = -EPS*v_hat/(sqrt(r_hat)+adam_eps);
							}
					}
			}
			layer[i]->update_W(update_W, update_b);
		}
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Update : %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif

#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
#ifdef USE_MPI
		if( UPDATE_ITER != -1 && (n+1) % UPDATE_ITER == 0 ){
			averaging();
		}
#endif
		
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Averaging : %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif
		each_func(*this, n+1, U[0], D);
	}

	for( int i = 0; i < num_layer; ++i ) layer[i]->finalize();	
}

#ifdef USE_GPU
template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::learning ( const std::vector<clMatrix<Real>>& X, const std::vector<clMatrix<Real>>& Y,
									  const int MAX_ITER,
									  const std::function<void(Neuralnet<clMatrix, Real>&, const int, const std::vector<clMatrix<Real>>&, const std::vector<clMatrix<Real>>&)>& each_func )
{
	int nprocs = 1, myrank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &myrank);
	MPI_Comm_size(inner_world, &nprocs);
#endif

	const int num_layer = layer.size();
	const int num_data = X[0].n;

	int seed = time(NULL);
#ifdef USE_MPI
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, inner_world);
#endif
	mt = std::mt19937(seed);

	// index of data
	std::vector<int> idx(num_data);
	std::iota(idx.begin(), idx.end(), 0);
	std::shuffle( idx.begin(), idx.end(), mt );

	// memory allocation for matrix U and D.
	std::vector<Mat<Real>> D(Y.size(), Mat<Real>(Y[0].m, BATCH_SIZE));
	std::vector<std::vector<Mat<Real>>> U(num_layer+1);
	U[0] = std::vector<Mat<Real>>(layer[0]->get_prev_num_map(), Mat<Real>(layer[0]->get_prev_num_unit(), BATCH_SIZE));
	for( int i = 0; i < U.size()-1; ++i ){
		U[i+1] = std::vector<Mat<Real>>(layer[i]->get_num_map(), Mat<Real>(layer[i]->get_num_unit(), BATCH_SIZE));
	}

	each_func(*this, 0, U[0], D);

	int cnt = 0;
	cl_int err;
	cl_mem cl_N, cl_idx, cl_offset;
	cl_N = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	cl_offset = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(int), NULL, &err);
	cl_idx = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, num_data*sizeof(int), NULL, &err);
	
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_N, CL_TRUE, 0, sizeof(int), &num_data, 0, NULL, NULL );
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_offset, CL_TRUE, 0, sizeof(int), &cnt, 0, NULL, NULL );
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_idx, CL_TRUE, 0, num_data*sizeof(int), &idx[0], 0, NULL, NULL );

	cl_mem cl_eps, cl_lambda;
	cl_eps = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(float), NULL, &err);
	cl_lambda = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(float), NULL, &err);	

	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_eps, CL_TRUE, 0, sizeof(float), &EPS, 0, NULL, NULL );
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_lambda, CL_TRUE, 0, sizeof(float), &LAMBDA, 0, NULL, NULL );

	cl_mem cl_adam_beta, cl_adam_gamma,
		cl_adam_beta_, cl_adam_gamma_, cl_adam_eps;
	cl_adam_beta = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(float), NULL, &err);
	cl_adam_gamma = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(float), NULL, &err);	
	cl_adam_beta_ = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(float), NULL, &err);
	cl_adam_gamma_ = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(float), NULL, &err);
	cl_adam_eps = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(float), NULL, &err);

	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_adam_beta, CL_TRUE, 0, sizeof(float), &adam_beta, 0, NULL, NULL );
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_adam_gamma, CL_TRUE, 0, sizeof(float), &adam_gamma, 0, NULL, NULL );
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_adam_eps, CL_TRUE, 0, sizeof(float), &adam_eps, 0, NULL, NULL );

	for( int n = 0; n < MAX_ITER; ++n ){
#ifdef DEBUG
		auto beg = std::chrono::system_clock::now();
#endif
		// assign data to mini-batch
		for( int i = 0; i < X.size(); ++i ){
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 0, &U[0][i].v );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 1, &X[i].v );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 2, &cl_idx );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 3, &cl_offset );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 4, &cl_N );
			cl_device_manager.run_kernel( PRG::ASSIGN_DATA, U[0][i].m, U[0][i].n );
		}
		for( int i = 0; i < Y.size(); ++i ){
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 0, &D[i].v );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 1, &Y[i].v );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 2, &cl_idx );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 3, &cl_offset );
			cl_device_manager.set_argument( PRG::ASSIGN_DATA, 4, &cl_N );
			cl_device_manager.run_kernel( PRG::ASSIGN_DATA, D[i].m, D[i].n );
		}
#ifdef DEBUG
		auto end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Init : %3lld %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count(), n);
#endif
#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
		// feed forward calculation
		for( int i = 0; i < num_layer; ++i ) {
#ifdef DEBUG
			auto beg = std::chrono::system_clock::now();
#endif
			auto V = U[i];
			if( i != 0 ){
				std::shared_ptr<Function<Real>> f = layer[i-1]->get_function();
				
				for( int j = 0; j < V.size(); ++j )
					V[j] = (*f)(V[j], false);
			}

			auto tmp = layer[i]->apply(V, false);
			for( int j = 0; j < tmp.size(); ++j ){
				U[i+1][j] = tmp[j];
			}
#ifdef DEBUG
			auto end = std::chrono::system_clock::now();
			if( myrank == 0 ) printf("  layer %d : %3lld\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif
		}
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Feed : %3lld %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count(), n);
#endif

		// back propagation calculation
#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
		std::vector<std::vector<std::vector<clMatrix<Real>>>> nabla_W, nabla_b;
		tie(nabla_W, nabla_b) = calc_gradient(U, D);
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Back : %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif

#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
		// averaging all gradients of weights of mini-batches
		for( int i = 0; i < nabla_W.size(); ++i )
			for( int j = 0; j < nabla_W[i].size(); ++j )
				for( int k = 0; k < nabla_W[i][j].size(); ++k )
					nabla_W[i][j][k] *= 1.0/BATCH_SIZE;
		for( int i = 0; i < nabla_b.size(); ++i )
			for( int j = 0; j < nabla_b[i].size(); ++j )
				for( int k = 0; k < nabla_b[i][j].size(); ++k )
					nabla_b[i][j][k] *= 1.0/BATCH_SIZE;
		
#ifdef CHECK_GRAD
		check_gradient(cnt, idx, X, Y, nabla_W, nabla_b, cl_N, cl_idx, cl_offset);
#endif
		cnt += BATCH_SIZE;
		if( cnt >= num_data ){
			std::shuffle( idx.begin(), idx.end(), mt );

			cnt = 0;
			clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_idx, CL_TRUE, 0, num_data*sizeof(int), &idx[0], 0, NULL, NULL );
		}
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_offset, CL_TRUE, 0, sizeof(int), &cnt, 0, NULL, NULL );

		// update W
		adam_beta_ *= adam_beta;
		adam_gamma_ *= adam_gamma;
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_adam_beta_, CL_TRUE, 0, sizeof(float), &adam_beta_, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_adam_gamma_, CL_TRUE, 0, sizeof(float), &adam_gamma_, 0, NULL, NULL );
		for( int i = 0; i < num_layer; ++i ){
			auto W = layer[i]->get_W();
			auto b = layer[i]->get_b();

			if( W.size() == 0 ) continue;

			std::vector<std::vector<clMatrix<Real>>> update_W(W.size(), std::vector<Mat<Real>>(W[0].size(), Mat<Real>(W[0][0].m, W[0][0].n))), update_b(b.size(), std::vector<Mat<Real>>(b[0].size(), Mat<Real>(b[0][0].m, b[0][0].n)));
			if( std::abs(LAMBDA) > 1.0E-15 ){
				for( int j = 0; j < W.size(); ++j )
					for( int k = 0; k < W[j].size(); ++k ){
						// cl_device_manager.set_argument( PRG::ADD_L2_REG, 0, &nabla_w[i][j][k].v );
						// cl_device_manager.set_argument( PRG::ADD_L2_REG, 1, &W[j][k].v );
						// cl_device_manager.set_argument( PRG::ADD_L2_REG, 2, &cl_lambda );
						// cl_device_manager.run_kernel( PRG::ADD_L2_REG, W[j][k].m, W[j][k].n-1 );
						nabla_W[i][j][k] += LAMBDA*W[j][k];
					}
			}

			for( int j = 0; j < nabla_W[i].size(); ++j )
				for( int k = 0; k < nabla_W[i][j].size(); ++k ){
					cl_device_manager.set_argument( PRG::ADAM, 0, &adam_v_W[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 1, &adam_r_W[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 2, &update_W[j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 3, &nabla_W[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 4, &cl_adam_beta );
					cl_device_manager.set_argument( PRG::ADAM, 5, &cl_adam_gamma );
					cl_device_manager.set_argument( PRG::ADAM, 6, &cl_adam_beta_ );
					cl_device_manager.set_argument( PRG::ADAM, 7, &cl_adam_gamma_ );
					cl_device_manager.set_argument( PRG::ADAM, 8, &cl_eps );
					cl_device_manager.set_argument( PRG::ADAM, 9, &cl_adam_eps );
					cl_device_manager.run_kernel( PRG::ADAM, W[j][k].m*W[j][k].n, 1 );
				}

			for( int j = 0; j < nabla_b[i].size(); ++j )
				for( int k = 0; k < nabla_b[i][j].size(); ++k ){
					cl_device_manager.set_argument( PRG::ADAM, 0, &adam_v_b[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 1, &adam_r_b[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 2, &update_b[j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 3, &nabla_b[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 4, &cl_adam_beta );
					cl_device_manager.set_argument( PRG::ADAM, 5, &cl_adam_gamma );
					cl_device_manager.set_argument( PRG::ADAM, 6, &cl_adam_beta_ );
					cl_device_manager.set_argument( PRG::ADAM, 7, &cl_adam_gamma_ );
					cl_device_manager.set_argument( PRG::ADAM, 8, &cl_eps );
					cl_device_manager.set_argument( PRG::ADAM, 9, &cl_adam_eps );
					cl_device_manager.run_kernel( PRG::ADAM, b[j][k].m*b[j][k].n, 1 );
				}
			layer[i]->update_W(update_W, update_b);
		}
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Update : %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif

#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
#ifdef USE_MPI
		if( UPDATE_ITER != -1 && (n+1) % UPDATE_ITER == 0 ){
			averaging();
		}
#endif
		
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Averaging : %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif
		each_func(*this, n+1, U[0], D);
	}

	for( int i = 0; i < num_layer; ++i ) layer[i]->finalize();	
}
#endif

template<template<typename> class Mat, typename Real>
std::vector<Mat<Real>> Neuralnet<Mat, Real>::apply ( const std::vector<Mat<Real>>& X ) const
{
	const int num_layer = layer.size();
	std::vector<Mat<Real>> U(X.size());
	for( int i = 0; i < X.size(); ++i ) U[i] = X[i];
	
	for( int i = 0; i < num_layer; ++i ){
		U = layer[i]->apply(U);
	}
	
	return U;
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::set_W ( const std::string& filename )
{
	for( int i = 0; i < layer.size(); ++i ){
		layer[i]->set_W("layer_" + std::to_string(i) + "_" + filename);
	}
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::output_W ( const std::string& filename ) const
{
	for( int i = 0; i < layer.size(); ++i ){
		layer[i]->output_W("layer_" + std::to_string(i) + "_" + filename);
	}
}

template<template<typename> class Mat, typename Real>
void Neuralnet<Mat, Real>::print_cost ( const std::vector<Mat<Real>>& x, const std::vector<Mat<Real>>& y ) const
{
	Real error[2] = { 0.0 };
	auto v = apply(x);

	error[0] = (*loss)(x, y, false);
	
	for( int i = 0; i < layer.size(); ++i ){
		auto W = layer[i]->get_W();
		for( int j = 0; j < W.size(); ++j )
			for( int k = 0; k < W[j].size(); ++k ){
				Real tmp = Mat<Real>::norm_fro(W[j][k]);
				error[1] += tmp*tmp;
			}
	}
	error[1] *= LAMBDA;

	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif

	if( rank == 0 ){
		printf("Cost     : Sum of costs  |   The cost    |L2 norm regul. |\n");
		printf("           %13.6E = %13.6E + %13.6E\n", error[0]+error[1], error[0], error[1]);
	}
}

// void Neuralnet::print_cost ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y ) const
// {
// #ifdef USE_GPU
// 	puts("WIP : print_cost");
// #else
// 	std::vector<Mat> X(x[0].size(), Mat(x[0][0].size(), x.size())), Y(y[0].size(), Mat(y[0][0].size(), y.size()));

// 	for( int i = 0; i < x[0].size(); ++i )
// 		for( int j = 0; j < x[0][0].size(); ++j )
// 			for( int k = 0; k < x.size(); ++k )
// 				X[i](j,k) = x[k][i][j];

// 	for( int i = 0; i < y[0].size(); ++i )
// 		for( int j = 0; j < y[0][0].size(); ++j )
// 			for( int k = 0; k < y.size(); ++k )
// 				Y[i](j,k) = y[k][i][j];
	
// 	print_cost( X, Y );
// #endif
// }

#endif
