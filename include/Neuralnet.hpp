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

#include "Layer.hpp"
#include "Function.hpp"
#include "Matrix.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

class Neuralnet
{
private:
	typedef float real;
#ifdef USE_GPU
	typedef clMatrix<real> Mat;
#else
	typedef Matrix<real> Mat;
#endif
	typedef std::vector<real> Vec;

	real adam_beta = 0.9, adam_gamma = 0.999, adam_eps = 1.0E-8;
	std::vector<std::vector<std::vector<Mat>>> adam_v, adam_r;
	real adam_beta_ = 1.0, adam_gamma_ = 1.0;

	int BATCH_SIZE, UPDATE_ITER;
	real EPS, LAMBDA;

	std::shared_ptr<LossFunction<real>> loss;
	std::vector<std::shared_ptr<Layer>> layer;
	
	std::mt19937 mt;
	std::uniform_real_distribution<real> d_rand;

#ifdef USE_MPI
	MPI_Comm outer_world, inner_world;
#endif

	std::vector<std::vector<std::vector<Mat>>> calc_gradient (const std::vector<std::vector<Mat>>& U, const std::vector<Mat>& d);
	void check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<Mat>& X, const std::vector<Mat>& Y, const std::vector<std::vector<std::vector<Mat>>>& nabla_w );
public:
	Neuralnet( const std::shared_ptr<LossFunction<real>>& loss );
#ifdef USE_MPI
	Neuralnet( const std::shared_ptr<LossFunction<real>>& loss, MPI_Comm outer_world, MPI_Comm inner_world );
#endif

	void set_EPS ( const real& EPS );
	void set_LAMBDA ( const real& LAMBDA );
	void set_BATCHSIZE ( const int& BATCH_SIZE );
	void set_UPDATEITER ( const int& UPDATE_ITER );

	void add_layer( const std::shared_ptr<Layer>& layer );

#ifdef USE_MPI
	void averaging ();
#endif
	void learning ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y,
					const int MAX_ITER = 1000,
					const std::function<void(Neuralnet&, const int, const std::vector<Mat>&, const std::vector<Mat>&)>& each_func
					= [](Neuralnet& nn, const int epoch, const std::vector<Mat>& x, const std::vector<Mat>& d) -> void {} );
	void learning ( const std::vector<Mat>& X, const std::vector<Mat>& Y,
					const int MAX_ITER = 1000,
					const std::function<void(Neuralnet&, const int, const std::vector<Mat>&, const std::vector<Mat>&)>& each_func
					= [](Neuralnet& nn, const int epoch, const std::vector<Mat>& x, const std::vector<Mat>& d) -> void {} );

	std::vector<Mat> apply ( const std::vector<Mat>& X ) const;
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& x ) const;

	void print_cost ( const std::vector<Mat>& x, const std::vector<Mat>& y ) const;
	void print_cost ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y ) const;
	void print_weight () const;
	void print_gradient () const;

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename ) const;
};

//////////////////// PRIVATE FUNCTION ////////////////////
std::vector<std::vector<std::vector<Neuralnet::Mat>>> Neuralnet::calc_gradient (
	const std::vector<std::vector<Mat>>& U, const std::vector<Mat>& d )
{
	const int num_layer = layer.size();
	
	std::vector<Mat> delta(d.size());
	for( int i = 0; i < d.size(); ++i ) delta[i] = Mat(d[i].m, d[i].n);

	std::shared_ptr<Function<real>> f = layer[num_layer-1]->get_function();
	for( int i = 0; i < d.size(); ++i ){
		auto tmp1 = (*f)(U[num_layer][i], false);
		auto tmp2 = (*f)(U[num_layer][i], true);
		auto tmp3 = (*loss)(tmp1, d[i], true);
		delta[i] = Mat::hadamard(tmp3, tmp2);
	}

#ifdef DEBUG
	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif
#endif
	std::vector<std::vector<std::vector<Mat>>> nabla_w(num_layer);
	for( int i = num_layer-1; i >= 0; --i ){
#ifdef DEBUG
		auto beg1 = std::chrono::system_clock::now();
#endif
		nabla_w[i] = layer[i]->calc_gradient(U[i], delta);
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
	return nabla_w;
}

void Neuralnet::check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<Mat>& X, const std::vector<Mat>& Y, const std::vector<std::vector<std::vector<Mat>>>& nabla_w )
{
	int rank = 0, target_rank = 0;
	int num_layer = this->layer.size();

#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif
	
	std::vector<Mat> tmp_X(X.size(), Mat(X[0].m, BATCH_SIZE)),
		tmp_Y(Y.size(), Mat(Y[0].m, BATCH_SIZE));
	for( int j = 0; j < X.size(); ++j ){
#ifdef USE_GPU
		auto tmp_input = X[j].get_matrix();
#else
		auto tmp_input = X[j];
#endif
		Matrix<real> tmp_output(X[0].m, BATCH_SIZE);
		for( int k = 0; k < X[0].m; ++k )
			for( int l = 0; l < BATCH_SIZE; ++l )
				tmp_output(k, l) = tmp_input(k, idx[cnt+l]);
		tmp_X[j] = tmp_output;
	}
	for( int j = 0; j < Y.size(); ++j ){
#ifdef USE_GPU
		auto tmp_input = Y[j].get_matrix();
#else
		auto tmp_input = Y[j];
#endif
		Matrix<real> tmp_output(Y[0].m, BATCH_SIZE);
		for( int k = 0; k < Y[0].m; ++k )
			for( int l = 0; l < BATCH_SIZE; ++l )
				tmp_output(k, l) = tmp_input(k, idx[cnt+l]);
		tmp_Y[j] = tmp_output;
	}

	// Calculate gradient numerically for confirmation of computing
	for( int i = 0; i < num_layer; ++i ){
		if( rank == target_rank ) printf("\tlayer %d\n", i);
		auto W = layer[i]->get_W();
		if( W.size() == 0 ) continue;

		int J = W.size(), K = W[0].size(), L = W[0][0].m, M = W[0][0].n;
#ifdef USE_MPI
		MPI_Bcast(&J, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&K, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&L, 1, MPI_INT, target_rank, inner_world);
		MPI_Bcast(&M, 1, MPI_INT, target_rank, inner_world);
#endif
		for( int j = 0; j < std::min(2, J); ++j ){ // num_map
			for( int k = 0; k < std::min(2, K); ++k ){ // prev_num_map
				for( int l = 0; l < std::min(2, L); ++l ){
					for( int m = 0; m < std::min(2, M); ++m ){
						double tmp;
						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
#ifdef USE_GPU
							tmp = W[j][k].get_element(l,m);
#else 
							tmp = W[j][k](l,m);
#endif
							tmp = 1.0E-3*(std::abs(tmp) < 1.0E0 ? 1.0 : std::abs(tmp));
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
#ifdef USE_GPU
							W[j][k].set_element(l,m,W[j][k].get_element(j,k)+tmp);
#else
							W[j][k](l,m) += tmp;
#endif
							layer[i]->set_W(W);
						}
						double E1 = 0.0;
						auto tmp1 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
#ifdef USE_GPU
							E1 += (*loss)(tmp1[n], tmp_Y[n], false).get_element(0, 0);
#else
							E1 += (*loss)(tmp1[n], tmp_Y[n], false)(0, 0);
#endif
						}

						if( layer[i]->get_num_map() != 1 || rank == target_rank ){
#ifdef USE_GPU
							W[j][k].set_element(l,m,W[j][k].get_element(j,k)-tmp);
#else
							W[j][k](l,m) -= tmp;
#endif
							layer[i]->set_W(W);
						}
						double E2 = 0.0;
						auto tmp2 = apply(tmp_X);
						for( int n = 0; n < Y.size(); ++n ){
#ifdef USE_GPU
							E2 += (*loss)(tmp2[n], tmp_Y[n], false).get_element(0, 0);
#else
							E2 += (*loss)(tmp2[n], tmp_Y[n], false)(0, 0);
#endif
						}
						
						if( rank == target_rank ){
#ifdef USE_GPU
							double grad = nabla_w[i][j][k].get_element(l,m);
#else
							double grad = nabla_w[i][j][k](l,m);
#endif

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

//////////////////// PUBLIC FUNCTION ////////////////////
Neuralnet::Neuralnet( const std::shared_ptr<LossFunction<real>>& loss )
	:EPS(1.0E-3), LAMBDA(0.0), BATCH_SIZE(1), UPDATE_ITER(-1), loss(loss)
{
	mt = std::mt19937(time(NULL));
}

#ifdef USE_MPI
Neuralnet::Neuralnet( const std::shared_ptr<LossFunction<real>>& loss, MPI_Comm outer_world, MPI_Comm inner_world )
	:EPS(1.0E-3), LAMBDA(0.0), BATCH_SIZE(1), UPDATE_ITER(-1), loss(loss), outer_world(outer_world), inner_world(inner_world)
{
	int rank = 0, seed;
	MPI_Comm_rank(outer_world, &rank);

	seed = time(NULL) + rank;
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, inner_world);
	mt = std::mt19937(seed);
}
#endif

void Neuralnet::set_EPS ( const real& EPS )
{
	this->EPS = EPS;
}

void Neuralnet::set_LAMBDA ( const real& LAMBDA )
{
	this->LAMBDA = LAMBDA;
}

void Neuralnet::set_BATCHSIZE ( const int& BATCH_SIZE )
{
	this->BATCH_SIZE = BATCH_SIZE;
}

void Neuralnet::set_UPDATEITER ( const int& UPDATE_ITER )
{
	this->UPDATE_ITER = UPDATE_ITER;
}

void Neuralnet::add_layer( const std::shared_ptr<Layer>& layer )
{
	std::shared_ptr<Function<real>> f;
	int prev_num_unit = -1, prev_num_map = -1;

	if( this->layer.size() == 0 )
		f = std::shared_ptr<Function<real>>(new Identity<real>);
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

	adam_v.push_back(std::vector<std::vector<Mat>>(w.size()));
	adam_r.push_back(std::vector<std::vector<Mat>>(w.size()));
	for( int j = 0; j < w.size(); ++j ){
		for( int k = 0; k < w[j].size(); ++k ){
			adam_v[idx][j].push_back(Mat::zeros(w[j][k].m, w[j][k].n));
			adam_r[idx][j].push_back(Mat::zeros(w[j][k].m, w[j][k].n));
		}
	}
}

#ifdef USE_MPI
void Neuralnet::averaging ()
{
	for( int i = 0; i < layer.size(); ++i ){
		layer[i]->param_mix();
		// for( int j = 0; j < adam_v[i].size(); ++j )
		// 	for( int k = 0; k < adam_v[i][j].size(); ++k )
		// 		for( int l = 0; l < adam_v[i][j][k].m; ++l )
		// 			for( int m = 0; m < adam_v[i][j][k].n; ++m ){
		// 				adam_v[i][j][k](l,m) = 0.0;
		// 				adam_r[i][j][k](l,m) = 0.0;
		// 			}
	}
	// adam_beta_ = adam_gamma_ = 1.0;
}
#endif

void Neuralnet::learning ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y,
						   const int MAX_ITER, const std::function<void(Neuralnet&, const int, const std::vector<Mat>&, const std::vector<Mat>&)>& each_func )
{
	const int num_layer = layer.size();
	const int num_data = x.size();
	const int num_dim_in = x[0][0].size();
	const int num_dim_out = y[0][0].size();

#ifdef USE_GPU
	puts("WIP : Neuralnet::learning");
#else
	std::vector<Mat> X(x[0].size(), Mat(num_dim_in, num_data)),
		Y(y[0].size(), Mat(num_dim_out, num_data));
	for( int i = 0; i < x[0].size(); ++i )
		for( int j = 0; j < num_dim_in; ++j )
			for( int k = 0; k < num_data; ++k )
				X[i](j,k) = x[k][i][j];
	for( int i = 0; i < y[0].size(); ++i )
		for( int j = 0; j < num_dim_out; ++j )
			for( int k = 0; k < num_data; ++k )
				Y[i](j,k) = y[k][i][j];
	learning(X, Y, MAX_ITER, each_func);
#endif
}

void Neuralnet::learning ( const std::vector<Mat>& X, const std::vector<Mat>& Y,
						   const int MAX_ITER,
						   const std::function<void(Neuralnet&, const int, const std::vector<Mat>&, const std::vector<Mat>&)>& each_func )
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
	std::vector<Mat> D(Y.size(), Mat(Y[0].m, BATCH_SIZE));
	std::vector<std::vector<Mat>> U(num_layer+1);
	U[0] = std::vector<Mat>(layer[0]->get_prev_num_map(), Mat(layer[0]->get_prev_num_unit(), BATCH_SIZE));
	for( int i = 0; i < U.size()-1; ++i ){
		U[i+1] = std::vector<Mat>(layer[i]->get_num_map(), Mat(layer[i]->get_num_unit(), BATCH_SIZE));
	}

	each_func(*this, 0, U[0], D);

	int cnt = 0;
#ifdef USE_GPU
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
#endif
	
	for( int n = 0; n < MAX_ITER; ++n ){
#ifdef DEBUG
		auto beg = std::chrono::system_clock::now();
#endif
		// assign data to mini-batch
#ifdef USE_GPU
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
#else
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
#endif
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
				std::shared_ptr<Function<real>> f = layer[i-1]->get_function();
				
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
		auto nabla_w = calc_gradient(U, D);
#ifdef DEBUG
		end = std::chrono::system_clock::now();
		if( myrank == 0 ) printf("Back : %3lld\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count());
#endif

#ifdef DEBUG
		beg = std::chrono::system_clock::now();
#endif
		// averaging all gradients of weights of mini-batches
		for( int i = 0; i < nabla_w.size(); ++i )
			for( int j = 0; j < nabla_w[i].size(); ++j )
				for( int k = 0; k < nabla_w[i][j].size(); ++k )
					nabla_w[i][j][k] *= 1.0/BATCH_SIZE;
		
#ifdef CHECK_GRAD
		check_gradient(cnt, idx, X, Y, nabla_w);
#endif
		cnt += BATCH_SIZE;
		if( cnt >= num_data ){
			std::shuffle( idx.begin(), idx.end(), mt );

			cnt = 0;
#ifdef USE_GPU
			clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_idx, CL_TRUE, 0, num_data*sizeof(int), &idx[0], 0, NULL, NULL );
#endif
		}
#ifdef USE_GPU
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_offset, CL_TRUE, 0, sizeof(int), &cnt, 0, NULL, NULL );
#endif

		// update W
		adam_beta_ *= adam_beta;
		adam_gamma_ *= adam_gamma;
#ifdef USE_GPU
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_adam_beta_, CL_TRUE, 0, sizeof(float), &adam_beta_, 0, NULL, NULL );
	clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_adam_gamma_, CL_TRUE, 0, sizeof(float), &adam_gamma_, 0, NULL, NULL );
#endif
		for( int i = 0; i < num_layer; ++i ){
			auto W = layer[i]->get_W();

			if( W.size() == 0 ) continue;

			std::vector<std::vector<Mat>> update_W(W.size(), std::vector<Mat>(W[0].size(), Mat(W[0][0].m, W[0][0].n)));
#ifdef USE_GPU
			if( std::abs(LAMBDA) > 1.0E-15 ){
				for( int j = 0; j < W.size(); ++j )
					for( int k = 0; k < W[j].size(); ++k ){
						cl_device_manager.set_argument( PRG::ADD_L2_REG, 0, &nabla_w[i][j][k].v );
						cl_device_manager.set_argument( PRG::ADD_L2_REG, 1, &W[j][k].v );
						cl_device_manager.set_argument( PRG::ADD_L2_REG, 2, &cl_lambda );
						cl_device_manager.run_kernel( PRG::ADD_L2_REG, W[j][k].m, W[j][k].n-1 );
					}
			}

			for( int j = 0; j < nabla_w[i].size(); ++j )
				for( int k = 0; k < nabla_w[i][j].size(); ++k ){
					cl_device_manager.set_argument( PRG::ADAM, 0, &adam_v[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 1, &adam_r[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 2, &update_W[j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 3, &nabla_w[i][j][k].v );
					cl_device_manager.set_argument( PRG::ADAM, 4, &cl_adam_beta );
					cl_device_manager.set_argument( PRG::ADAM, 5, &cl_adam_gamma );
					cl_device_manager.set_argument( PRG::ADAM, 6, &cl_adam_beta_ );
					cl_device_manager.set_argument( PRG::ADAM, 7, &cl_adam_gamma_ );
					cl_device_manager.set_argument( PRG::ADAM, 8, &cl_eps );
					cl_device_manager.set_argument( PRG::ADAM, 9, &cl_adam_eps );
					cl_device_manager.run_kernel( PRG::ADAM, W[j][k].m*W[j][k].n, 1 );
				}
#else
#pragma omp parallel
			{
				if( std::abs(LAMBDA) > 1.0E-15 ){
					// L2 norm regularization
					for( int j = 0; j < W.size(); ++j )
						for( int k = 0; k < W[j].size(); ++k )
#pragma omp for schedule(auto) nowait
							for( int l = 0; l < W[j][k].m; ++l )
								for( int m = 1; m < W[j][k].n; ++m )
									nabla_w[i][j][k](l,m) += LAMBDA*W[j][k](l,m);
				}

				// ADAM
				for( int j = 0; j < nabla_w[i].size(); ++j )
					for( int k = 0; k < nabla_w[i][j].size(); ++k )
#pragma omp for schedule(auto) nowait
						for( int l = 0; l < nabla_w[i][j][k].m; ++l )
							for( int m = 0; m < nabla_w[i][j][k].n; ++m ){
								adam_v[i][j][k](l,m) = adam_beta*adam_v[i][j][k](l,m) + (1.0 - adam_beta)*nabla_w[i][j][k](l,m);
								adam_r[i][j][k](l,m) = adam_gamma*adam_r[i][j][k](l,m) + (1.0 - adam_gamma)*(nabla_w[i][j][k](l,m)*nabla_w[i][j][k](l,m));
							}

#pragma omp barrier
				
				for( int j = 0; j < W.size(); ++j )
					for( int k = 0; k < W[j].size(); ++k ){
#pragma omp for schedule(auto) nowait
						for( int l = 0; l < update_W[j][k].m; ++l )
							for( int m = 0; m < update_W[j][k].n; ++m ){
								auto v_hat = adam_v[i][j][k](l,m) / (1.0 - adam_beta_);
								auto r_hat = adam_r[i][j][k](l,m) / (1.0 - adam_gamma_);
								update_W[j][k](l,m) = -EPS*v_hat/(sqrt(r_hat)+adam_eps);
							}
					}
			}
#endif
			layer[i]->update_W(update_W);
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

#ifdef USE_GPU
	clReleaseMemObject( cl_N );
	clReleaseMemObject( cl_idx );
	clReleaseMemObject( cl_offset );

	clReleaseMemObject( cl_eps );
	clReleaseMemObject( cl_lambda );

	clReleaseMemObject( cl_adam_beta );
	clReleaseMemObject( cl_adam_gamma );
	clReleaseMemObject( cl_adam_beta_ );
	clReleaseMemObject( cl_adam_gamma_ );
	clReleaseMemObject( cl_adam_eps );
#endif
	
	for( int i = 0; i < num_layer; ++i ) layer[i]->finalize();	
}

std::vector<Neuralnet::Mat> Neuralnet::apply ( const std::vector<Mat>& X ) const
{
	const int num_layer = layer.size();
	std::vector<Mat> U(X.size());
	for( int i = 0; i < X.size(); ++i ) U[i] = X[i];
	
	for( int i = 0; i < num_layer; ++i ){
		U = layer[i]->apply(U);
	}
	
	return U;
}

std::vector<std::vector<Neuralnet::Vec>> Neuralnet::apply ( const std::vector<std::vector<Vec>>& x ) const
{
#ifdef USE_GPU
	puts("WIP : Neuralnet::apply");
#else
	std::vector<Mat> u(x[0].size());
	for( int i = 0; i < x[0].size(); ++i ) u[i] = Mat(x[0][0].size(), x.size());
	for( int i = 0; i < x.size(); ++i )
		for( int j = 0; j < x[0].size(); ++j )
			for( int k = 0; k < x[0][0].size(); ++k )
				u[j](k,i) = x[i][j][k];

	u = apply(u);

	std::vector<std::vector<Vec>> ret(u[0].n);
	for( int i = 0; i < u[0].n; ++i ){
		ret[i] = std::vector<Vec>(u.size(), Vec(u[0].m));
		for( int j = 0; j < u.size(); ++j )
			for( int k = 0; k < u[0].m; ++k )
				ret[i][j][k] = u[j](k,i);
	}
	return ret;
#endif
}

void Neuralnet::set_W ( const std::string& filename )
{
	for( int i = 0; i < layer.size(); ++i ){
		layer[i]->set_W("layer_" + std::to_string(i) + "_" + filename);
	}
}

void Neuralnet::output_W ( const std::string& filename ) const
{
	for( int i = 0; i < layer.size(); ++i ){
		layer[i]->output_W("layer_" + std::to_string(i) + "_" + filename);
	}
}

void Neuralnet::print_cost ( const std::vector<Mat>& x, const std::vector<Mat>& y ) const
{
	double error[3] = { 0.0 }, min_err = 1.0E100, max_err = 0.0;
	auto v = apply(x);
	for( int i = 0; i < x.size(); ++i ){
		for( int j = 0; j < x[i].n; ++j ){
			Mat v_ = v[i].sub(0, j, v[i].m, 1), y_ = y[i].sub(0, j, y[i].m, 1);
#ifdef USE_GPU
			double sum = (*loss)(v_, y_, false).get_element(0,0);
#else
			double sum = (*loss)(v_, y_, false)(0,0);
#endif
		
			min_err = std::min(min_err, sum);
			max_err = std::max(max_err, sum);
			error[0] += sum;
		}
	}
	error[0] /= x[0].n;

	for( int i = 0; i < layer.size(); ++i ){
		auto W = layer[i]->get_W();
		for( int j = 0; j < W.size(); ++j )
			for( int k = 0; k < W[j].size(); ++k ){
				real tmp = Mat::norm_fro(W[j][k].sub(0, 1, W[j][k].m, W[j][k].n-1));
				error[2] += tmp*tmp;
			}
	}
	error[2] *= LAMBDA;

	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif

	if( rank == 0 ){
		printf("Cost     :    Average    |      Min      |      Max      |\n");
		printf("           %13.6E | %13.6E | %13.6E |\n", error[0], min_err, max_err);
		printf("           Sum of costs  |   The cost    |L2 norm regul. |\n");
		printf("           %13.6E = %13.6E + %13.6E\n",
			   error[0]+error[1]+error[2], error[0], error[2]);
	}
}

void Neuralnet::print_cost ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y ) const
{
#ifdef USE_GPU
	puts("WIP : print_cost");
#else
	std::vector<Mat> X(x[0].size(), Mat(x[0][0].size(), x.size())), Y(y[0].size(), Mat(y[0][0].size(), y.size()));

	for( int i = 0; i < x[0].size(); ++i )
		for( int j = 0; j < x[0][0].size(); ++j )
			for( int k = 0; k < x.size(); ++k )
				X[i](j,k) = x[k][i][j];

	for( int i = 0; i < y[0].size(); ++i )
		for( int j = 0; j < y[0][0].size(); ++j )
			for( int k = 0; k < y.size(); ++k )
				Y[i](j,k) = y[k][i][j];
	
	print_cost( X, Y );
#endif
}

void Neuralnet::print_weight () const
{
	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif

	if( rank == 0 ) printf("Weight   :    Average    |      Min      |      Max      |\n");
	for( int i = 0; i < layer.size(); ++i ){
		real ave_weight = 0.0;
		real max_weight = -1.0E100;
		real min_weight = 1.0E100;

		int num = 0;
		auto W = layer[i]->get_W();
		for( int j = 0; j < W.size(); ++j ){
			for( int k = 0; k < W[j].size(); ++k ){
#ifdef USE_GPU
				auto tmp_W = W[j][k].get_matrix();
#else
				auto tmp_W = W[j][k];
#endif
				for( int l = 0; l < W[j][k].m; ++l )
					for( int m = 0; m < W[j][k].n; ++m ){
						auto tmp = std::abs(tmp_W(k,m));

						ave_weight += tmp;
						max_weight = std::max(max_weight, tmp);
						min_weight = std::min(min_weight, tmp);
					}
				num += W[j][k].m*W[j][k].n;
			}
		}
		ave_weight /= num;

		if( rank == 0 ){
			if( W.size() == 0 )
				printf(" Layer %d   ------------- | ------------- | ------------- |\n", i);
			else
				printf(" Layer %d   %13.6E | %13.6E | %13.6E |\n", i, ave_weight, min_weight, max_weight);
		}
	}
}

void Neuralnet::print_gradient () const
{
	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif

#ifdef USE_GPU
	puts("WIP : print_gradient");
#else
	if( rank == 0 ) printf("Gradient :    Average    |      Min      |      Max      |\n");
	for( int i = 0; i < layer.size(); ++i ){
		double ave_gradient = 0.0;
		double max_gradient = -1.0E100;
		double min_gradient = 1.0E100;

		int num = 0;
		auto W = layer[i]->get_W();
		for( int j = 0; j < W.size(); ++j ){
			for( int k = 0; k < W[j].size(); ++k ){
				for( int l = 0; l < W[j][k].m; ++l )
					for( int m = 0; m < W[j][k].n; ++m ){
						auto v_hat = adam_v[i][j][k](l,m)/(1.0 - adam_beta_);
						auto r_hat = adam_r[i][j][k](l,m)/(1.0 - adam_gamma_);
						auto tmp = std::abs(-EPS*v_hat/(sqrt(r_hat) + adam_eps));
						
						ave_gradient += tmp;
						max_gradient = std::max(max_gradient, tmp);
						min_gradient = std::min(min_gradient, tmp);
					}
				num += W[j][k].m*W[j][k].n;
			}
		}
		ave_gradient /= num;

		if( rank == 0 ){
			if( W.size() == 0 )
				printf(" Layer %d   ------------- | ------------- | ------------- |\n", i);
			else
				printf(" Layer %d   %13.6E | %13.6E | %13.6E |\n", i, ave_gradient, min_gradient, max_gradient);
		}
	}
#endif
}

#endif
