#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <random>

#include "Layer.hpp"
#include "Function.hpp"
#include "matrix.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

class Neuralnet
{
private:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	const double adam_beta = 0.9, adam_gamma = 0.999, adam_eps = 1.0E-8;
	std::vector<std::vector<std::vector<Mat>>> adam_v, adam_r;
	double adam_beta_ = 1.0, adam_gamma_ = 1.0;

	int BATCH_SIZE;
	double EPS, LAMBDA;

	std::shared_ptr<LossFunction> loss;
	std::vector<std::shared_ptr<Layer>> layer;
	
	std::mt19937 m;
	std::uniform_real_distribution<double> d_rand;

	std::vector<std::vector<std::vector<Mat>>> calc_gradient (const std::vector<std::vector<Mat>>& U, const std::vector<Mat>& d);
	void check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y, const std::vector<std::vector<std::vector<Mat>>>& nabla_w );
public:
	Neuralnet( const std::shared_ptr<LossFunction>& loss );

	void set_EPS ( const double& EPS );
	void set_LAMBDA ( const double& LAMBDA );
	void set_BATCHSIZE ( const int& BATCH_SIZE );

	void add_layer( const std::shared_ptr<Layer>& layer );

	void learning ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y,
					const int MAX_ITER = 1000,
					const std::function<void(const Neuralnet&, const int, const std::vector<Mat>&, const std::vector<Mat>&)>& each_func
					= [](const Neuralnet& nn, const int epoch, const std::vector<Mat>& x, const std::vector<Mat>& d) -> void {} );

	std::vector<Mat> apply ( const std::vector<Mat>& X ) const;
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& x ) const;

	void print_cost ( const std::vector<Mat>& x, const std::vector<Mat>& y ) const;
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

	std::shared_ptr<Function> f = layer[num_layer-1]->get_function();
	for( int i = 0; i < d.size(); ++i )
		delta[i] = Mat::hadamard((*loss)((*f)(U[num_layer][i], false), d[i], true),
								 (*f)(U[num_layer][i], true));

	std::vector<std::vector<std::vector<Mat>>> nabla_w(num_layer);
	for( int i = num_layer-1; i >= 0; --i ){
		nabla_w[i] = layer[i]->calc_gradient(U[i], delta);
		
		if( i == 0 ) continue;
		delta = layer[i]->calc_delta(U[i], delta);
	}
	return nabla_w;
}

void Neuralnet::check_gradient ( int cnt, const std::vector<int>& idx, const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y, const std::vector<std::vector<std::vector<Mat>>>& nabla_w )
{
	int num_layer = this->layer.size();

	// Calculate gradient numerically for confirmation of computing
	for( int i = 0; i < num_layer; ++i ){
		printf("\tlayer %d\n", i);
		auto W = layer[i]->get_W();
		std::vector<std::vector<Vec>> X(BATCH_SIZE);
		for( int j = 0; j < BATCH_SIZE; ++j ) X[j] = x[idx[cnt+j]];
		for( int j = 0; j < std::min(2, (int)W.size()); ++j ){ // num_map
			for( int k = 0; k < std::min(2, (int)W[j].size()); ++k ){ // prev_num_map
				for( int l = 0; l < std::min(5, (int)W[j][k].m); ++l ){
					for( int m = 0; m < std::min(5, (int)W[j][k].n); ++m ){
						auto tmp = 1.0E-6*(std::abs(W[j][k](l,m)) < 1.0E-3 ? 1.0 : std::abs(W[j][k](l,m)));;

						W[j][k](l,m) += tmp;
						layer[i]->set_W(W);
						double E1 = 0.0;
						auto tmp1 = apply(X);
						for( int n = 0; n < tmp1[0].size(); ++n )
							for( int o = 0; o < BATCH_SIZE; ++o ){
								E1 += (*loss)(Mat(tmp1[o][n]), Mat(y[idx[cnt+o]][n]), false)(0,0);
							}
						W[j][k](l,m) -= tmp;
						layer[i]->set_W(W);
						double E2 = 0.0;
						auto tmp2 = apply(X);
						for( int n = 0; n < tmp2[0].size(); ++n )
							for( int o = 0; o < BATCH_SIZE; ++o )
								E2 += (*loss)(Mat(tmp2[o][n]), Mat(y[idx[cnt+o]][n]), false)(0,0);

						printf("\t%3d, %3d, %3d, %3d : ( %.10E, %.10E = %.10E )\n", j, k, l, m, 0.5*(E1 - E2)/tmp/BATCH_SIZE, nabla_w[i][j][k](l,m), (std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE - nabla_w[i][j][k](l,m)))/std::abs(0.5*(E1 - E2)/tmp/BATCH_SIZE));
					}
				}
			}
		}
		puts("");
	}
}

//////////////////// PUBLIC FUNCTION ////////////////////
Neuralnet::Neuralnet( const std::shared_ptr<LossFunction>& loss )
	:EPS(1.0E-3), LAMBDA(1.0E-5), BATCH_SIZE(1), loss(loss)
{
	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	m = std::mt19937(time(NULL) + rank);
}

void Neuralnet::set_EPS ( const double& EPS )
{
	this->EPS = EPS;
}

void Neuralnet::set_LAMBDA ( const double& LAMBDA )
{
	this->LAMBDA = LAMBDA;
}

void Neuralnet::set_BATCHSIZE ( const int& BATCH_SIZE )
{
	this->BATCH_SIZE = BATCH_SIZE;
}

void Neuralnet::add_layer( const std::shared_ptr<Layer>& layer )
{
	std::shared_ptr<Function> f;
	if( this->layer.size() == 0 )
		f = std::shared_ptr<Function>(new Identity);
	else
		f = this->layer[this->layer.size()-1]->get_function();
	
	this->layer.emplace_back( layer );

	int idx = this->layer.size()-1;
	this->layer[idx]->set_prev_function(f);
	this->layer[idx]->init(m);

	auto w = layer->get_W();

	adam_v.push_back(std::vector<std::vector<Mat>>(w.size()));
	adam_r.push_back(std::vector<std::vector<Mat>>(w.size()));
	for( int j = 0; j < w.size(); ++j ){
		for( int k = 0; k < w[j].size(); ++k ){
			adam_v[idx][j].emplace_back(w[j][k].m, w[j][k].n);
			adam_r[idx][j].emplace_back(w[j][k].m, w[j][k].n);
		}
	}
}

void Neuralnet::learning ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y,
						   const int MAX_ITER, const std::function<void(const Neuralnet&, const int, const std::vector<Mat>&, const std::vector<Mat>&)>& each_func )
{
	int nprocs = 1, myrank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif
	const int num_layer = layer.size();

	std::vector<int> idx(x.size() / nprocs);
	iota(idx.begin(), idx.end(), x.size()/nprocs*myrank);
	shuffle( idx.begin(), idx.end(), m );

	int cnt = 0;
	for( int n = 0; n <= MAX_ITER; ++n ){
		std::vector<Mat> D;
		std::vector<std::vector<Mat>> U(num_layer+1);

		for( int i = 0; i < x[0].size(); ++i ){
			U[0].emplace_back(x[0][i].size(), BATCH_SIZE);
			for( int j = 0; j < U[0][i].m; ++j )
				for( int k = 0; k < BATCH_SIZE; ++k )
					U[0][i](j,k) = x[idx[cnt+k]][i][j];
		}
		
		for( int i = 0; i < y[0].size(); ++i ){
			D.emplace_back(y[0][i].size(), BATCH_SIZE);
			for( int j = 0; j < D[i].m; ++j )
				for( int k = 0; k < BATCH_SIZE; ++k )
					D[i](j,k) = y[idx[cnt+k]][i][j];
		}

		for( int i = 0; i < num_layer; ++i ) {
			auto V = U[i];
			if( i != 0 ){
				std::shared_ptr<Function> f = layer[i-1]->get_function();
				
				for( int j = 0; j < V.size(); ++j )
					V[j] = (*f)(V[j], false);
			}
			auto tmp = layer[i]->apply(V, false);
			
			for( int j = 0; j < tmp.size(); ++j ){
				U[i+1].emplace_back(tmp[j].m, tmp[j].n);
				U[i+1][j] = tmp[j];
			}
		}

		auto nabla_w = calc_gradient(U, D);
		
		for( int i = 0; i < nabla_w.size(); ++i )
			for( int j = 0; j < nabla_w[i].size(); ++j )
				for( int k = 0; k < nabla_w[i][j].size(); ++k )
					nabla_w[i][j][k] = 1.0/BATCH_SIZE * nabla_w[i][j][k];
		
		// check_gradient(cnt, idx, x, y, nabla_w);
		cnt += BATCH_SIZE;
		if( cnt >= x.size()/nprocs ){
			shuffle( idx.begin(), idx.end(), m );
			cnt = 0;
		}

		// update W
		for( int i = 0; i < num_layer; ++i ){
			// L2 norm regularization
			auto W = layer[i]->get_W();
			
			if( W.size() == 0 ) continue;
			for( int j = 0; j < W.size(); ++j )
				for( int k = 0; k < W[j].size(); ++k )
					for( int l = 0; l < W[j][k].m; ++l )
						for( int m = 1; m < W[j][k].n; ++m )
							nabla_w[i][j][k](l,m) += LAMBDA*W[j][k](l,m);

			// ADAM
			adam_beta_ *= adam_beta;
			adam_gamma_ *= adam_gamma;
			for( int j = 0; j < W.size(); ++j )
				for( int k = 0; k < W[j].size(); ++k )
					for( int l = 0; l < W[j][k].m; ++l )
						for( int m = 0; m < W[j][k].n; ++m ){
							adam_v[i][j][k](l,m) = adam_beta*adam_v[i][j][k](l,m) + (1.0 - adam_beta)*nabla_w[i][j][k](l,m);
							adam_r[i][j][k](l,m) = adam_gamma*adam_r[i][j][k](l,m) + (1.0 - adam_gamma)*(nabla_w[i][j][k](l,m)*nabla_w[i][j][k](l,m));
						}

			std::vector<std::vector<Mat>> update_W(W.size(), std::vector<Mat>(W[0].size()));
			for( int j = 0; j < W.size(); ++j )
				for( int k = 0; k < W[j].size(); ++k ){
					update_W[j][k] = Mat::zeros(W[j][k].m, W[j][k].n);
					for( int l = 0; l < W[j][k].m; ++l )
						for( int m = 0; m < W[j][k].n; ++m ){
							auto v_hat = adam_v[i][j][k](l,m) / (1.0 - adam_beta_);
							auto r_hat = adam_r[i][j][k](l,m) / (1.0 - adam_gamma_);
							update_W[j][k](l,m) = -EPS*v_hat/(sqrt(r_hat)+adam_eps);
						}
				}

			layer[i]->update_W(update_W);
		}

		each_func(*this, n, U[0], D);
	}

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

	std::vector<Mat> ret(U.size());
	for( int i = 0; i < U.size(); ++i ){
		ret[i] = Mat::zeros(U[i].m, U[i].n);
		for( int j = 0; j < U[i].m; ++j )
			for( int k = 0; k < U[i].n; ++k )
				ret[i](j,k) = U[i](j,k);
	}
	
	return ret;
}

std::vector<std::vector<Neuralnet::Vec>> Neuralnet::apply ( const std::vector<std::vector<Vec>>& x ) const
{
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
			Mat v_(v[i].m, 1), y_(y[i].m, 1);
			for( int k = 0; k < v[i].m; ++k ){
				v_(k,0) = v[i](k,j);
				y_(k,0) = y[i](k,j);
			}
			double sum = (*loss)(v_, y_, false)(0,0);
		
			min_err = std::min(min_err, sum);
			max_err = std::max(max_err, sum);
			error[0] += sum;
		}
	}
	error[0] /= x[0].n;

	for( int i = 0; i < layer.size(); ++i ){
		auto W = layer[i]->get_W();
		for( int j = 0; j < W.size(); ++j )
			for( int k = 0; k < W[j].size(); ++k )
				for( int l = 0; l < W[j][k].m; ++l )
					for( int m = 1; m < W[j][k].n; ++m )
						error[2] += W[j][k](l,m)*W[j][k](l,m);
	}
	error[2] *= LAMBDA;

	printf("Cost     :    Average    |      Min      |      Max      |\n");
	printf("           %13.6E | %13.6E | %13.6E |\n", error[0], min_err, max_err);
	printf("           Sum of costs  |   The cost    |L2 norm regul. |\n");
	printf("           %13.6E = %13.6E + %13.6E\n",
		   error[0]+error[1]+error[2], error[0], error[2]);
}

void Neuralnet::print_weight () const
{
	printf("Weight   :    Average    |      Min      |      Max      |\n");
	for( int i = 0; i < layer.size(); ++i ){
		double ave_weight = 0.0;
		double max_weight = -1.0E100;
		double min_weight = 1.0E100;

		int num = 0;
		auto W = layer[i]->get_W();
		for( int j = 0; j < W.size(); ++j ){
			for( int k = 0; k < W[j].size(); ++k ){
				for( int l = 0; l < W[j][k].m; ++l )
					for( int m = 0; m < W[j][k].n; ++m ){
						auto tmp = std::abs(W[j][k](k,m));

						ave_weight += tmp;
						max_weight = std::max(max_weight, tmp);
						min_weight = std::min(min_weight, tmp);
					}
				num += W[j][k].m*W[j][k].n;
			}
		}
		ave_weight /= num;

		if( W.size() == 0 )
			printf(" Layer %d   ------------- | ------------- | ------------- |\n", i);
		else
			printf(" Layer %d   %13.6E | %13.6E | %13.6E |\n", i, ave_weight, min_weight, max_weight);
	}
}

void Neuralnet::print_gradient () const
{
	printf("Gradient :    Average    |      Min      |      Max      |\n");
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

		if( W.size() == 0 )
			printf(" Layer %d   ------------- | ------------- | ------------- |\n", i);
		else
			printf(" Layer %d   %13.6E | %13.6E | %13.6E |\n", i, ave_gradient, min_gradient, max_gradient);
	}
}

#endif
