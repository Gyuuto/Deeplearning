#ifndef NEURALNET_HPP
#define NEURALNET_HPP

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

#include "matrix.hpp"

class Neuralnet
{
private:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	int BATCH_SIZE;
	double EPS, LAMBDA, MU, BETA, RHO, K, P;

	int num_layer;
	std::vector<int> num_unit;
	std::vector<Mat> W;
	
	std::vector<std::function<double(double)>> activate_func;
	std::vector<std::function<double(double)>> activate_diff_func;

	std::mt19937 m;
	std::uniform_real_distribution<double> d_rand;

	std::vector<Mat> calc_gradient (const std::vector<Mat>& u, const Mat& d);
public:
	Neuralnet( const std::vector<int>& num_unit );
	void set_function ( const int& idx,
						const std::function<double(double)>& func,
						const std::function<double(double)>& diff_func );
	void set_EPS ( const double& EPS );
	void set_LAMBDA ( const double& LAMBDA );
	void set_MU ( const double& MU );
	void set_BETA ( const double& BETA );
	void set_RHO ( const double& RHO );
	void set_K ( const double& K );
	void set_P ( const double& P );
	void set_BATCHSIZE ( const int& BATCH_SIZE );

	void set_W ( const std::string& filename );
	void set_W ( int idx, const Mat& w );
	Mat get_W ( int idx );
	
	void learning ( const std::vector<Vec>& x, const std::vector<Vec>& y,
					const int MAX_ITER = 1000 );

	Mat apply ( const Mat& X );
	Vec apply ( const Vec& x );

	void output_W ( const std::string& filename );
};

//////////////////// PRIVATE FUNCTION ////////////////////
std::vector<Neuralnet::Mat> Neuralnet::calc_gradient (
	const std::vector<Mat>& u, const Mat& d )
{
	Mat delta(d.m, d.n);
	for( int i = 0; i < delta.m; ++i ){
		for( int j = 0; j < delta.n; ++j ){
			delta[i][j] = (activate_func[num_layer-1](u[num_layer-1][i+1][j]) - d[i][j]) *
				activate_diff_func[num_layer-1](u[num_layer-1][i+1][j]);
		}
	}
	
	std::vector<Mat> nabla_w(W.size());
	for( int i = 0; i < W.size(); ++i ) nabla_w[i] = Mat::zeros(W[i].m, W[i].n);
	for( int i = num_layer-2; i >= 0; --i ){
		for( int j = 0; j < nabla_w[i].m; ++j ){
			for( int k = 0; k < nabla_w[i].n; ++k ){
				for( int l = 0; l < delta.n; ++l ){
					nabla_w[i][j][k] += delta[j][l]*(
						i == 0 || k == 0 ? u[i][k][l] : activate_func[i-1](u[i][k][l])
						);
				}
			}
		}
		
		if( i == 0 ) continue;
		auto W_T = Matrix<double>::transpose(W[i]);
		auto nx_delta = W_T*delta;
		delta = Mat(nx_delta.m-1, nx_delta.n);
		for( int j = 0; j < delta.m; ++j ){
			for( int k = 0; k < delta.n; ++k ){
				delta[j][k] = nx_delta[j+1][k]*activate_diff_func[i-1](u[i][j+1][k]);
			}
		}
	}
	
	return nabla_w;
}

//////////////////// PUBLIC FUNCTION ////////////////////
Neuralnet::Neuralnet( const std::vector<int>& num_unit )
	:num_layer(num_unit.size()), num_unit(num_unit), EPS(1.0E-1), LAMBDA(1.0E-5), MU(0.3), BETA(0.0), RHO(0.0), BATCH_SIZE(1), K(1.0)
{
	m = std::mt19937(time(NULL));
	for( int i = 1; i < this->num_layer; ++i ){
		double tmp = sqrt(6.0/(num_unit[i] + num_unit[i-1]));
		d_rand = std::uniform_real_distribution<double>(-tmp, tmp);
		W.push_back( Mat(num_unit[i], num_unit[i-1]+1) );

		auto& w = *W.rbegin();
		for( int j = 0; j < w.m; ++j )
			for( int k = 0; k < w.n; ++k )
				w[j][k] = d_rand(m);
	}

	activate_func.resize( this->num_layer );
	activate_diff_func.resize( this->num_layer );
}

void Neuralnet::set_function ( const int& idx,
								  const std::function<double(double)>& func,
								  const std::function<double(double)>& diff_func )
{
	activate_func[idx] = func;
	activate_diff_func[idx] = diff_func;
}

void Neuralnet::set_EPS ( const double& EPS )
{
	this->EPS = EPS;
}

void Neuralnet::set_LAMBDA ( const double& LAMBDA )
{
	this->LAMBDA = LAMBDA;
}

void Neuralnet::set_MU ( const double& MU )
{
	this->MU = MU;
}
	
void Neuralnet::set_BETA ( const double& BETA )
{
	this->BETA = BETA;
}

void Neuralnet::set_RHO ( const double& RHO )
{
	this->RHO = RHO;
}

void Neuralnet::set_BATCHSIZE ( const int& BATCH_SIZE )
{
	this->BATCH_SIZE = BATCH_SIZE;
}

void Neuralnet::set_K ( const double& K )
{
	this->K = K;
}

void Neuralnet::set_P ( const double& P )
{
	this->P = P;
}

void Neuralnet::set_W ( const std::string& filename )
{
	std::ifstream input(filename, std::ios_base::in | std::ios_base::binary);

	int idx = 0;
	while( !input.eof() ){
		int M, N;
		input.read((char*)&M, sizeof(M));
		input.read((char*)&N, sizeof(N));

		W[idx] = Mat(M, N);
		for( int i = 0; i < M; ++i ){
			for( int j = 0; j < N; ++j ){
				input.read((char*)&W[idx][i][j], sizeof(W[idx][i][j]));
			}
		}
		++idx;
		if( idx == num_layer-1 ) break;
	}
}

void Neuralnet::set_W ( int idx, const Mat& w )
{
	W[idx] = w;
}

Neuralnet::Mat Neuralnet::get_W ( int idx )
{
	return W[idx];
}

void Neuralnet::learning ( const std::vector<Vec>& x, const std::vector<Vec>& y, const int MAX_ITER )
{
	std::vector<int> idx(x.size());
	iota(idx.begin(), idx.end(), 0);

	int cnt = 0;
	std::vector<Mat> prev_W(num_layer-1);
	std::vector<Mat> ada_grad(num_layer-1);
	for( int i = 0; i < num_layer-1; ++i ){
		prev_W[i] = Mat(W[i].m, W[i].n);
		ada_grad[i] = Mat(W[i].m, W[i].n);
	}
	for( int n = 0; n <= MAX_ITER; ++n ){
		Mat D(y[0].size(), BATCH_SIZE);
		std::vector<Mat> U(num_layer);
		U[0] = Mat(x[0].size()+1, BATCH_SIZE);
		for( int i = 0; i < BATCH_SIZE; ++i ){
			U[0][0][i] = 1.0;
			for( int j = 0; j < x[idx[cnt+i]].size(); ++j ) U[0][j+1][i] = x[idx[cnt+i]][j];
		}

		for( int i = 0; i < D.m; ++i )
			for( int j = 0; j < D.n; ++j )
				D[i][j] = y[idx[cnt+j]][i];
		
		for( int i = 1; i < num_layer; ++i ) {
			auto V = U[i-1];
			if( i != 1 )
				for( int j = 1; j < V.m; ++j ) for( int k = 0; k < V.n; ++k )
						V[j][k] = activate_func[i-2](V[j][k]);
			
			auto tmp = W[i-1]*V;
			
			U[i] = Mat(tmp.m+1, BATCH_SIZE);
			for( int j = 0; j < U[i].n; ++j ) U[i][0][j] = 1.0;
			for( int j = 0; j < tmp.m; ++j ) for( int k = 0; k < tmp.n; ++k )
					U[i][j+1][k] = tmp[j][k];

			if( !(std::abs(1.0 - K) < 1.0E-10) && i != num_layer-1 ){
				std::vector<int> idx(num_unit[i]);
				std::iota( idx.begin(), idx.end(), 0 );
				
				for( int j = 0; j < BATCH_SIZE; ++j ){
					auto comp = [&]( const int& i1, const int& i2 ) -> bool {
						return activate_func[i-1](U[i][i1+1][j]) > activate_func[i-1](U[i][i2+1][j]);
					};

					sort( idx.begin(), idx.end(), comp );

					for( int k = num_unit[i]*K; k < num_unit[i]; ++k ) U[i][1+idx[k]][j] = 0.0;
				}
			}
		}

		auto nabla_w = calc_gradient(U, D);
		for( int i = 0; i < nabla_w.size(); ++i ) nabla_w[i] = 1.0/BATCH_SIZE * nabla_w[i];

		// Calculate gradient numerically for confirmation of computing
		// BATCH_SIZE = 1 is required!!
		// for( int i = 0; i < nabla_w.size(); ++i ){
		// 	printf("\tlayer %d\n", i);
		// 	for( int j = 0; j < nabla_w[i].m; ++j ){
		// 		for( int k = 0; k < nabla_w[i].n; ++k ){
		// 			const double EPS = 1.0E-6*std::abs(W[i][j][k]);
					
		// 			W[i][j][k] += EPS;
		// 			auto tmp1 = Mat(apply(x[cnt]));
		// 			auto E1 = (Mat::transpose(tmp1 - Mat(y[cnt]))*(tmp1 - Mat(y[cnt])))[0][0];
					
		// 			W[i][j][k] -= EPS;
		// 			auto tmp2 = Mat(apply(x[cnt]));
		// 			auto E2 = (Mat::transpose(tmp2 - Mat(y[cnt]))*(tmp2 - Mat(y[cnt])))[0][0];
					
		// 			printf("\t%3d, %3d : ( %.10E, %.10E = %.10E)\n", j, k, 0.5*(E1 - E2)/EPS, nabla_w[i][j][k], std::abs(0.5*(E1 - E2)/EPS - nabla_w[i][j][k]));
		// 			// tmp_nabla_w[j][k][l] = 0.5*(E1[0][0] - E2[0][0])/EPS;
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
		for( int i = 0; i < num_layer-1; ++i ){
			auto update_W = W[i];
			for( int j = 0; j < update_W.m; ++j ) update_W[j][0] = 0.0;

			// AdaGrad
			for( int j = 0; j < W[i].m; ++j ){
				double tmp = 0.0;
				for( int k = 0; k < W[i].n; ++k )
					ada_grad[i][j][k] += nabla_w[i][j][k]*nabla_w[i][j][k];
			}

			for( int j = 0; j < update_W.m; ++j )
				for( int k = 0; k < update_W.n; ++k )
					update_W[j][k] = EPS/(1.0+sqrt(ada_grad[i][j][k]))*(LAMBDA*update_W[j][k] - nabla_w[i][j][k]) + MU*prev_W[i][j][k];
			
			W[i] = W[i] + update_W;
			prev_W[i] = update_W;
		}

		if( n%(x.size()/BATCH_SIZE) == 0 ){
			printf("%lu Epoch : \n", n/(x.size()/BATCH_SIZE));

			// puts("Gradient of W :");
			// for( int i = 0; i < num_layer-1; ++i ){
			// 	printf("layer %d : \n", i);
			// 	for( int j = 0; j < nabla_w[i].m; ++j ){
			// 		printf("\t");
			// 		for( int k = 0; k < nabla_w[i].n; ++k ) printf("%.3E ", nabla_w[i][j][k]);
			// 		puts("");
			// 	}
			// }

			// puts("W :");
			// for( int i = 0; i < num_layer-1; ++i ){
			// 	printf("layer %d : \n", i);
			// 	for( int j = 0; j < W[i].m; ++j ){
			// 		printf("\t");
			// 		for( int k = 0; k < W[i].n; ++k ) printf("%.3E ", W[i][j][k]);
			// 		puts("");
			// 	}
			// }
			
			double error = 0.0, min_err = 1.0E100, max_err = 0.0;
			for( int i = 0; i < x.size(); ++i ){
				Vec v = apply(x[i]);
				double sum = 0.0;
				for( int j = 0; j < v.size(); ++j )
					sum += (v[j] - y[i][j])*(v[j] - y[i][j]);
				min_err = std::min(min_err, sum);
				max_err = std::max(max_err, sum);
				error += sum;
			}
			error *= 0.5;

			printf("Error : %.6E\nAverage : %.6E\nMin : %.6E\nMax : %.6E\n", error, error/x.size(), min_err, max_err);
			printf("Learning Rate : \n");
			for( int i = 0; i < num_layer-1; ++i ){
				double learn_rate = 0.0;
				double max_learn_rate = EPS/sqrt(1.0 + ada_grad[i][0][0]);
				double min_learn_rate = EPS/sqrt(1.0 + ada_grad[i][0][0]);
				for( int j = 0; j < ada_grad[i].m; ++j )
					for( int k = 0; k < ada_grad[i].n; ++k ){
						learn_rate += EPS/sqrt(1.0 + ada_grad[i][j][k]);
						max_learn_rate = std::max(max_learn_rate, EPS/(1.0 + sqrt(ada_grad[i][j][k])));
						min_learn_rate = std::min(min_learn_rate, EPS/(1.0 + sqrt(ada_grad[i][j][k])));
					}
				printf("\tLayer %d : (%.6E, %.6E, %.6E)\n", i, learn_rate / (ada_grad[i].m*ada_grad[i].n), max_learn_rate, min_learn_rate);
			}
			puts("");
			fflush(stdout);
			// for( int i = 0; i < num_layer-1; ++i ) ada_grad[i] *= 0.8;
		}
	}
}

Neuralnet::Mat Neuralnet::apply ( const Mat& X )
{
	Mat U(X.m+1, X.n);

	for( int i = 0; i < X.n; ++i ){
		U[0][i] = 1.0;
		for( int j = 0; j < X.m; ++j ) U[j+1][i] = X[j][i];
	}
	for( int i = 0; i < num_layer-1; ++i ){
		auto V = W[i]*U;

		if( i == num_layer-2 ){
			U = Mat(V.m, V.n);
			for( int j = 0; j < V.m; ++j )
				for( int k = 0; k < V.n; ++k )
					U[j][k] = activate_func[i](V[j][k]);
			continue;
		}
		U = Mat(V.m+1, V.n);
		
		for( int j = 0; j < V.n; ++j ){
			U[0][j] = 1.0;
			for( int k = 0; k < V.m; ++k ) U[k+1][j] = activate_func[i](V[k][j]);
		}
	}

	return U;
}

Neuralnet::Vec Neuralnet::apply ( const Vec& x )
{
	Mat u(x.size()+1, 1);

	u[0][0] = 1.0;
	for( int i = 0; i < x.size(); ++i ) u[i+1][0] = x[i];
	for( int i = 0; i < num_layer-1; ++i ){
		auto v = W[i]*u;

		if( i == num_layer-2 ){
			u = Mat(v.m, 1);
			for( int j = 0; j < v.m; ++j ) u[j][0] = activate_func[i](v[j][0]);
			continue;
		}
		u = Mat(v.m+1, 1);
		u[0][0] = 1.0;
		for( int j = 0; j < v.m; ++j ) u[j+1][0] = activate_func[i](v[j][0]);
	}

	Vec ret(u.m);
	for( int i = 0; i < u.m; ++i ) ret[i] = u[i][0];

	return ret;
}

void Neuralnet::output_W ( const std::string& filename )
{
	std::ofstream ofs(filename, std::ios_base::out | std::ios_base::binary);
	for( int i = 0; i < num_layer-1; ++i ){
		ofs.write((char*)&W[i].m, sizeof(W[i].m));
		ofs.write((char*)&W[i].n, sizeof(W[i].n));
		for( int j = 0; j < W[i].m; ++j )
			for( int k = 0; k < W[i].n; ++k )
				ofs.write((char*)&W[i][j][k], sizeof(W[i][j][k]));
	}	
}

#endif
