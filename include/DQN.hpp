#include <vector>
#include <algorithm>
#include <functional>
#include <memory>
#include <random>
#include <string>

#include "Layer.hpp"
#include "Function.hpp"
#include "matrix.hpp"

#include "Neuralnet.hpp"

struct DQN_Memory
{
	typedef std::vector<double> Vec;
	std::vector<Vec> s, s_n;
	int a_id;
	double r;

	DQN_Memory() {}
	DQN_Memory ( const std::vector<Vec>& s, const std::vector<Vec>& s_n, const int a_id, const double r )
		: s(s), s_n(s_n), a_id(a_id), r(r)
	{}
};

struct State
{
	typedef std::vector<double> Vec;
	std::vector<Vec> s;
	double r;
};

class MaxSquare : public LossFunction
{
public:
	inline Matrix<double> operator() ( const Matrix<double>& x, const Matrix<double>& d, const bool& isdiff ){
		if( isdiff ){
			Matrix<double> y = Matrix<double>::zeros(x.m, x.n);

			for( int i = 0; i < x.n; ++i ){
				int id = 0;
				double val = x(0,i);
				for( int j = 1; j < x.m; ++j )
					if( val < x(j,i) ){
						val = x(j,i);
						id = j;
					}
				y(id,i) = val - d(id,i);
			}
			
			return y;
		}
		else{
			Matrix<double> y = Matrix<double>::zeros(x.m, x.n);

			for( int i = 0; i < x.n; ++i ){
				int id = 0;
				double val = x(0,i);
				for( int j = 1; j < x.m; ++j )
					if( val < x(j,i) ){
						val = x(j,i);
						id = j;
					}
				y(id,i) = (val - d(id,i))*(val - d(id,i));
			}
			
			return y;
		}
	}
};

class DQN
{
private:
	typedef Matrix<double> Mat;
	typedef std::vector<double> Vec;

	int mem_capacity, max_id;
	double alpha, gamma, epsilon;
	std::mt19937 mt;
	std::uniform_int_distribution<int> i_rand;
	std::uniform_real_distribution<double> d_rand;

	std::function<State(int)> func_trans; // that is returning next state and reward when give a input action id
	std::vector<DQN_Memory> mem;
	std::vector<std::shared_ptr<Layer>> layers;
	Neuralnet *Q, *Q_tilde;
	
public:
	DQN ( const int mem_capacity, const int max_id,
		  const double alpha, const double gamma, const double epsilon,
		  const std::function<State(int)>& func_trans, const std::vector<std::shared_ptr<Layer>>& layers );
	~DQN ();
	
	int get_next_action ( const std::vector<Vec>& state );

	void learning( const int max_iter, const int batch_size, const int C );
};

DQN::DQN(const int mem_capacity, const int max_id,
		 const double alpha, const double gamma, const double epsilon,
		 const std::function<State(int)>& func_trans, const std::vector<std::shared_ptr<Layer>>& layers )
	: mem_capacity(mem_capacity), max_id(max_id),
	  alpha(alpha), gamma(gamma), epsilon(epsilon),
	  func_trans(func_trans), layers(layers)
{
	mt = std::mt19937(time(NULL));
	i_rand = std::uniform_int_distribution<int>(0, max_id-1);
	d_rand = std::uniform_real_distribution<double>(0.0, 1.0);
	
	mem.resize(mem_capacity);

	Q = new Neuralnet(std::shared_ptr<LossFunction>(new MaxSquare));
	Q_tilde = new Neuralnet(std::shared_ptr<LossFunction>(new MaxSquare));
	for( int i = 0; i < this->layers.size(); ++i ){
		Q->add_layer(this->layers[i]);
		Q_tilde->add_layer(this->layers[i]);
	}
}

DQN::~DQN()
{
	delete Q;
	delete Q_tilde;
}

int DQN::get_next_action ( const std::vector<Vec>& state )
{
	std::vector<std::vector<Vec>> a = Q->apply(std::vector<std::vector<Vec>>(1, state));
	int id = 0;
	double max_Q = a[0][0][0];

	for( int i = 1; i < a[0][0].size(); ++i ){
		if( max_Q < a[0][0][i] ){
			id = i;
			max_Q = a[0][0][i];
		}
	}
	
	return id;
}

void DQN::learning( const int max_iter, const int batch_size, const int C )
{
	int mem_id = 0;
	
	Q->set_EPS(alpha);
	Q->set_LAMBDA(0.0);
	Q->set_BATCHSIZE(batch_size);

	State cur_state = func_trans(-1);
	for( int n = 0; n < max_iter; ++n ){
		int next_act_id = -1;
		
		if( mem_id < mem_capacity || d_rand(mt) - epsilon < 0.0 ){ // random action
			next_act_id = i_rand(mt);
		}
		else{
			next_act_id = get_next_action(cur_state.s);
		}

		printf("Next act id : %d\n", next_act_id);
		State next_state = func_trans(next_act_id);

		mem[mem_id%mem_capacity] = DQN_Memory(cur_state.s, next_state.s, next_act_id, next_state.r);
		++mem_id;

		// Is this needed to generate x and d in each loop?
		const int data_size = std::min(mem_id, mem_capacity);
		const int dim = next_state.s.size();
		if( data_size > batch_size ){
			std::vector<std::vector<Vec>> x(data_size), d(data_size);
			for( int i = 0; i < data_size; ++i ){
				x[i] = mem[i].s;

				auto tmp = Q_tilde->apply(std::vector<std::vector<Vec>>(1, mem[i].s_n));
				int id = 0;
				double val = tmp[0][0][0];
				for( int j = 1; j < max_id; ++j ){
					if( val < tmp[0][0][j] ){
						id = j;
						val = tmp[0][0][j];
					}
				}
				d[i] = std::vector<Vec>(1, Vec(max_id, 0.0));
				d[i][0][id] = next_state.r + gamma * val;
			}

			for( int i = 0; i < max_id; ++i ){
				printf("%d:%.3E ", i, d[0][0][i]);
			}puts("");

			// do back propagation with mem
			Q->learning(x, d, data_size/batch_size*C,
						[&](Neuralnet& nn, const int iter, const std::vector<Mat>& x, const std::vector<Mat>& d) -> void{
							if( iter == (data_size / batch_size)*C ){
								printf("Iter : %d\n", n);
								nn.print_cost(x, d);
							}
						});
		}
		if( n % 100 == 0 ){
			std::string str = "W_" + std::to_string(n) + ".dat";
			Q->output_W(str);
		}
		cur_state = next_state;
	}
}
