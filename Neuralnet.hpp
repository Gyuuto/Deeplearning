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
#include "matrix.hpp"

class Neuralnet
{
private:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	int BATCH_SIZE;
	double EPS, LAMBDA;

	std::vector<std::shared_ptr<Layer>> layer;
	
	std::mt19937 m;
	std::uniform_real_distribution<double> d_rand;

	std::vector<std::vector<Mat>> calc_gradient (const std::vector<std::vector<Mat>>& U, const std::vector<Mat>& d);
public:
	Neuralnet();

	void set_EPS ( const double& EPS );
	void set_LAMBDA ( const double& LAMBDA );
	void set_BATCHSIZE ( const int& BATCH_SIZE );

	void add_layer( const std::shared_ptr<Layer>& layer );

	void learning ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y,
					const int MAX_ITER = 1000 );

	std::vector<Mat> apply ( const std::vector<Mat>& X );
	std::vector<Vec> apply ( const std::vector<Vec>& x );

	void output_W ( const std::string& filename );
};

//////////////////// PRIVATE FUNCTION ////////////////////
std::vector<std::vector<Neuralnet::Mat>> Neuralnet::calc_gradient (
	const std::vector<std::vector<Mat>>& U, const std::vector<Mat>& d )
{
	const int num_layer = layer.size();
	
	std::vector<Mat> delta(d.size());
	for( int i = 0; i < d.size(); ++i ) delta[i] = Mat(d[i].m, d[i].n);
	for( int i = 0; i < d.size(); ++i )
		for( int j = 0; j < d[i].m; ++j )
			for( int k = 0; k < d[i].n; ++k ){
				std::function<double(double)> f, d_f;
				tie(f, d_f) = layer[num_layer-1]->get_function();
				
				delta[i][j][k] = (f(U[num_layer][i][j][k]) - d[i][j][k]) *
					d_f(U[num_layer][i][j][k]);
			}
	
	std::vector<std::vector<Mat>> nabla_w(num_layer);
	for( int i = num_layer-1; i >= 0; --i ){
		std::function<double(double)> f, d_f;
		if( i != 0 ) tie(f, d_f) = layer[i-1]->get_function();
		else{ f = [](const double& x) -> double { return x; }; d_f = f; }
		nabla_w[i] = layer[i]->calc_gradient(U[i], delta, f);
		delta = layer[i]->calc_delta(U[i], delta, d_f);
	}

	return nabla_w;
}

//////////////////// PUBLIC FUNCTION ////////////////////
Neuralnet::Neuralnet()
	:EPS(1.0E-1), LAMBDA(1.0E-5), BATCH_SIZE(1)
{
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
	this->layer.emplace_back( layer );
}

void Neuralnet::learning ( const std::vector<std::vector<Vec>>& x, const std::vector<std::vector<Vec>>& y, const int MAX_ITER )
{
	const int num_layer = layer.size();

	std::vector<int> idx(x.size());
	iota(idx.begin(), idx.end(), 0);

	int cnt = 0;
	std::vector<std::vector<Mat>> adam_v(num_layer), adam_r(num_layer);
	const double adam_beta = 0.9, adam_gamma = 0.999, adam_eps = 1.0E-8;
	double adam_beta_ = 1.0, adam_gamma_ = 1.0;
	for( int i = 0; i < num_layer; ++i ){
		auto w_i = layer[i]->get_W();
		for( int j = 0; j < w_i.size(); ++j ){
			adam_v[i].emplace_back(w_i[j].m, w_i[j].n);
			adam_r[i].emplace_back(w_i[j].m, w_i[j].n);
		}
	}
	for( int n = 0; n <= MAX_ITER; ++n ){
		std::vector<Mat> D;
		std::vector<std::vector<Mat>> U(num_layer+1);
		for( int i = 0; i < x[0].size(); ++i ){
			U[0].emplace_back(x[0][i].size(), BATCH_SIZE);
			for( int j = 0; j < BATCH_SIZE; ++j )
				for( int k = 0; k < x[idx[cnt+j]][i].size(); ++k )
					U[0][i][k][j] = x[idx[cnt+j]][i][k];
		}

		for( int i = 0; i < y[0].size(); ++i ){
			D.emplace_back(y[0][i].size(), BATCH_SIZE);
			for( int j = 0; j < D[i].m; ++j )
				for( int k = 0; k < D[i].n; ++k )
					D[i][j][k] = y[idx[cnt+k]][i][j];
		}
		
		for( int i = 0; i < num_layer; ++i ) {
			auto V = U[i];
			if( i != 0 ){
				std::function<double(double)> f, d_f;
				tie(f, d_f) = layer[i-1]->get_function();
				for( int j = 0; j < V.size(); ++j )
					for( int k = 0; k < V[j].m; ++k )
						for( int l = 0; l < V[j].n; ++l )
							V[j][k][l] = f(V[j][k][l]);
			}
			auto tmp = layer[i]->apply(V, false);
			
			for( int j = 0; j < tmp.size(); ++j ){
				U[i+1].emplace_back(tmp[j].m, tmp[j].n);
				for( int k = 0; k < tmp[j].m; ++k ){
					for( int l = 0; l < tmp[j].n; ++l )
						U[i+1][j][k][l] = tmp[j][k][l];
				}
			}
		}
		
		auto nabla_w = calc_gradient(U, D);
		
		for( int i = 0; i < nabla_w.size(); ++i )
			for( int j = 0; j < nabla_w[i].size(); ++j )
				nabla_w[i][j] = 1.0/BATCH_SIZE * nabla_w[i][j];
		
		// Calculate gradient numerically for confirmation of computing
		// BATCH_SIZE = 1 is required!!
		// for( int i = 0; i < nabla_w.size(); ++i ){
		// 	printf("\tlayer %d\n", i);
		// 	auto W = layer[i]->get_W();
		// 	for( int j = 0; j < nabla_w[i].size(); ++j ){
		// 		for( int k = 0; k < nabla_w[i][j].m; ++k ){
		// 			for( int l = 0; l < nabla_w[i][j].n; ++l ){
		// 				auto tmp = 1.0E-8*(std::abs(W[j][k][l]) < 1.0E-4 ? 1.0 : std::abs(W[j][k][l]));;
		// 				std::vector<Mat> EPS;
		// 				EPS.emplace_back(nabla_w[i][j].m, nabla_w[i][j].n);
		// 				EPS[0][k][l] = tmp;
						
		// 				layer[i]->update_W(EPS);
		// 				auto tmp1 = Mat(apply(x[cnt])[0]);
		// 				auto E1 = (Mat::transpose(tmp1 - Mat(y[cnt][0]))*(tmp1 - Mat(y[cnt][0])))[0][0];

		// 				EPS[0][k][l] = -1.0*EPS[0][k][l];
		// 				layer[i]->update_W(EPS);
		// 				auto tmp2 = Mat(apply(x[cnt])[0]);
		// 				auto E2 = (Mat::transpose(tmp2 - Mat(y[cnt][0]))*(tmp2 - Mat(y[cnt][0])))[0][0];
						
		// 				printf("\t%3d, %3d : ( %.10E, %.10E = %.10E)\n", k, l, 0.5*(E1 - E2)/tmp, nabla_w[i][j][k][l], (std::abs(0.5*(E1 - E2)/tmp - nabla_w[i][j][k][l]))/std::abs(0.5*(E1 - E2)/tmp));
		// 				// nabla_w[i][j][k][l] = 0.5*(E1 - E2)/tmp;
		// 			}
		// 		}
		// 	}
		// 	puts("");
		// }
		cnt += BATCH_SIZE;
		if( cnt >= x.size() ){
			shuffle( idx.begin(), idx.end(), m );
			cnt %= x.size();
		}
		
		// update W
		for( int i = 0; i < num_layer; ++i ){
			// L2 norm regularization
			auto W = layer[i]->get_W();
			for( int j = 0; j < W.size(); ++j )
				for( int k = 0; k < W[j].m; ++k )
					for( int l = 1; l < W[j].n; ++l )
						nabla_w[i][j][k][l] += LAMBDA*W[j][k][l];

			// ADAM
			adam_beta_ *= adam_beta;
			adam_gamma_ *= adam_gamma;
			for( int j = 0; j < W.size(); ++j )
				for( int k = 0; k < W[j].m; ++k )
					for( int l = 0; l < W[j].n; ++l ){
						adam_v[i][j][k][l] = adam_beta*adam_v[i][j][k][l] + (1.0 - adam_beta)*nabla_w[i][j][k][l];
						adam_r[i][j][k][l] = adam_gamma*adam_r[i][j][k][l] + (1.0 - adam_gamma)*(nabla_w[i][j][k][l]*nabla_w[i][j][k][l]);
					}

			std::vector<Mat> update_W(W.size());
			for( int j = 0; j < W.size(); ++j ){
				update_W[j] = Mat::zeros(W[j].m, W[j].n);
				for( int k = 0; k < W[j].m; ++k )
					for( int l = 0; l < W[j].n; ++l ){
						auto v_hat = adam_v[i][j][k][l] / (1.0 - adam_beta_);
						auto r_hat = adam_r[i][j][k][l] / (1.0 - adam_gamma_);
						update_W[j][k][l] = -EPS*v_hat/(sqrt(r_hat)+adam_eps);
				}
			}

			layer[i]->update_W(update_W);
		}

		if( n%(x.size()/BATCH_SIZE) == 0 ){
			printf("%lu Epoch : \n", n/(x.size()/BATCH_SIZE));
			
			double error[3] = { 0.0 }, min_err = 1.0E100, max_err = 0.0;
			for( int i = 0; i < x.size(); ++i ){
				std::vector<Vec> v = apply(x[i]);
				double sum = 0.0;
				for( int j = 0; j < v.size(); ++j )
					for( int k = 0; k < v[j].size(); ++k )
						sum += (v[j][k] - y[i][j][k])*(v[j][k] - y[i][j][k]);
				min_err = std::min(min_err, sum);
				max_err = std::max(max_err, sum);
				error[0] += sum;
			}
			error[0] /= x.size();

			for( int i = 0; i < num_layer; ++i ){
				auto W = layer[i]->get_W();
				for( int j = 0; j < W.size(); ++j )
					for( int k = 0; k < W[j].m; ++k )
						for( int l = 1; l < W[j].n; ++l )
							error[2] += W[j][k][l]*W[j][k][l];
			}
			error[2] *= LAMBDA;
			
			printf("Error    :    Average    |      Min      |      Max      |\n");
			printf("           %13.6E | %13.6E | %13.6E |\n", error[0], min_err, max_err);
			printf("           Sum of errors | Squared error | Sparse regul. |L2 norm regul. |\n");
			printf("           %13.6E = %13.6E + %13.6E + %13.6E\n",
				   error[0]+error[1]+error[2], error[0], error[1], error[2]);
			printf("Gradient :    Average    |      Min      |      Max      |\n");
			for( int i = 0; i < num_layer; ++i ){
				double ave_gradient = 0.0;
				double max_gradient = -1.0E100;
				double min_gradient = 1.0E100;

				int num = 0;
				auto W = layer[i]->get_W();
				for( int j = 0; j < W.size(); ++j ){
					for( int k = 0; k < W[j].m; ++k )
						for( int l = 0; l < W[j].n; ++l ){
							auto v_hat = adam_v[i][j][k][l]/(1.0 - adam_beta_);
							auto r_hat = adam_r[i][j][k][l]/(1.0 - adam_gamma_);
							auto tmp = -EPS*v_hat/(sqrt(r_hat) + adam_eps);
						
							ave_gradient += tmp;
							max_gradient = std::max(max_gradient, tmp);
							min_gradient = std::min(min_gradient, tmp);
						}
					num += W[j].m*W[j].n;
				}
				ave_gradient /= num;

				printf(" Layer %d   %13.6E | %13.6E | %13.6E |\n", i, ave_gradient, min_gradient, max_gradient);
			}
			puts("");
			fflush(stdout);

			output_W("autoenc_W.dat");
		}
	}
}

std::vector<Neuralnet::Mat> Neuralnet::apply ( const std::vector<Mat>& X )
{
	const int num_layer = layer.size();
	std::vector<Mat> U(X.size());
	for( int i = 0; i < X.size(); ++i ) U[i] = Mat(X[i].m, X[i].n);

	for( int i = 0; i < X.size(); ++i )
		for( int j = 0; j < X[i].n; ++j )
			for( int k = 0; k < X[i].m; ++k ) U[i][k][j] = X[i][k][j];
	
	for( int i = 0; i < num_layer; ++i ){
		std::function<double(double)> f, d_f;
		tie(f, d_f) = layer[i]->get_function();
		auto V = layer[i]->apply(U);

		U = std::vector<Mat>(V.size());
		for( int j = 0; j < V.size(); ++j ){
			U[j] = Mat::zeros( V[j].m, V[j].n );
			for( int k = 0; k < V[j].n; ++k )
				for( int l = 0; l < V[j].m; ++l ) U[j][l][k] = f(V[j][l][k]);
		}
	}

	std::vector<Mat> ret(X.size());
	for( int i = 0; i < X.size(); ++i ){
		ret[i] = Mat::zeros(U[i].m, U[i].n);
		for( int j = 0; j < U[i].m; ++j )
			for( int k = 0; k < U[i].n; ++k )
				ret[i][j][k] = U[i][j][k];
	}
	
	return ret;
}

std::vector<Neuralnet::Vec> Neuralnet::apply ( const std::vector<Vec>& x )
{
	std::vector<Mat> u(x.size());
	for( int i = 0; i < x.size(); ++i ) u[i] = Mat(x[i]);
	u = apply(u);
	
	std::vector<Vec> ret(x.size());
	for( int i = 0; i < x.size(); ++i ){
		ret[i] = Vec(u[i].m);
		for( int j = 0; j < u[i].m; ++j )
			ret[i][j] = u[i][j][0];
	}
	
	return ret;
}

void Neuralnet::output_W ( const std::string& filename )
{
	for( int i = 0; i < layer.size(); ++i ){
		layer[i]->output_W("layer_" + std::to_string(i) + filename);
	}
}

#endif
