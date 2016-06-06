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

#ifdef USE_MPI
#include <mpi.h>
#endif

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

enum DQN_STATE
{
	EPISODE_CONT = 0,
	EPISODE_END
};

struct State
{
	typedef std::vector<double> Vec;
	std::vector<Vec> s;
	double r;
	DQN_STATE ep_s;
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
			Matrix<double> y = Matrix<double>::zeros(1, 1);

			for( int i = 0; i < x.n; ++i ){
				int id = 0;
				double val = x(0,i);
				for( int j = 1; j < x.m; ++j )
					if( val < x(j,i) ){
						val = x(j,i);
						id = j;
					}
				y(0,0) += (val - d(id,i))*(val - d(id,i));
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

	int mem_capacity, mem_id, max_id;
	int update_model_freq, initial_mem;
	double alpha, gamma, epsilon;
	std::mt19937 mt;
	std::uniform_int_distribution<int> i_rand;
	std::uniform_real_distribution<double> d_rand;

	std::function<State(int)> func_trans; // that is returning next state and reward when give a input action id
	std::function<bool(int)> func_act; // that is returning possibility given action

	std::vector<DQN_Memory> mem;
	std::vector<std::shared_ptr<Layer>> layers, tilde_layers;

	Neuralnet *Q, *Q_tilde;

#ifdef USE_MPI
	MPI_Comm inner_world, outer_world;
#endif
public:
	DQN ( const int mem_capacity, const int initial_mem, const int max_id,
		  const std::function<State(int)>& func_trans, const std::function<bool(int)>& func_act,
		  const std::vector<std::shared_ptr<Layer>>& layers, const std::vector<std::shared_ptr<Layer>>& tilde_layers
#ifdef USE_MPI
		  , MPI_Comm outer_world, MPI_Comm inner_world
#endif
		);
	~DQN ();

	void set_update_model_freq ( int update_model_freq );
	void set_alpha ( double alpha );
	void set_gamma ( double gamma );
	void set_epsilon ( double epsilon );
	
	int get_next_action ( const std::vector<Vec>& state );

	void learning( const int max_iter, const int batch_size );

	void output_W ( const std::string& filename );
	void set_W ( const std::string& filename );
};

DQN::DQN(const int mem_capacity, const int initial_mem, const int max_id,
		 const std::function<State(int)>& func_trans, const std::function<bool(int)>& func_act,
		 const std::vector<std::shared_ptr<Layer>>& layers, const std::vector<std::shared_ptr<Layer>>& tilde_layers
#ifdef USE_MPI
		 , MPI_Comm outer_world, MPI_Comm inner_world 
#endif
	)
	: mem_capacity(mem_capacity), initial_mem(initial_mem), mem_id(0), max_id(max_id),
	alpha(1.0E-3), gamma(0.95), epsilon(0.05), update_model_freq(1000),
	func_trans(func_trans), func_act(func_act), layers(layers), tilde_layers(tilde_layers)
{
	int seed = time(NULL);
	
#ifdef USE_MPI
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, inner_world);
#endif
	mt = std::mt19937(seed);
	i_rand = std::uniform_int_distribution<int>(0, max_id-1);
	d_rand = std::uniform_real_distribution<double>(0.0, 1.0);
	
	mem.resize(mem_capacity);

#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;
	Q = new Neuralnet(std::shared_ptr<LossFunction>(new MaxSquare), outer_world, inner_world);
	Q_tilde = new Neuralnet(std::shared_ptr<LossFunction>(new MaxSquare), outer_world, inner_world);
#else
	Q = new Neuralnet(std::shared_ptr<LossFunction>(new MaxSquare));
	Q_tilde = new Neuralnet(std::shared_ptr<LossFunction>(new MaxSquare));
#endif

	for( int i = 0; i < this->layers.size(); ++i ){
		Q->add_layer(this->layers[i]);
		Q_tilde->add_layer(this->tilde_layers[i]);
	}
}
	
DQN::~DQN()
{
	delete Q;
	delete Q_tilde;
}

void DQN::set_update_model_freq ( int update_model_freq )
{
	this->update_model_freq = update_model_freq;
}

void DQN::set_alpha ( double alpha )
{
	this->alpha = alpha;
}

void DQN::set_gamma ( double gamma )
{
	this->gamma = gamma;
}

void DQN::set_epsilon ( double epsilon )
{
	this->epsilon = epsilon;
}

int DQN::get_next_action ( const std::vector<Vec>& state )
{
	std::vector<std::vector<Vec>> a = Q->apply(std::vector<std::vector<Vec>>(1, state));
	int id = -100;
	double max_Q = -1.0E100;

	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	for( int i = 0; i < max_id; ++i ){
		if( func_act(i) && max_Q < a[0][0][i] ){
			id = i;
			max_Q = a[0][0][i];
		}
	}
	
	return id;
}

void DQN::learning( const int max_iter, const int batch_size )
{
	Q->set_EPS(alpha);
	Q->set_LAMBDA(0.0);
	Q->set_BATCHSIZE(batch_size);

	double eps = 1.0;
	State cur_state = func_trans(-1);

	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(inner_world, &rank);
#endif
	for( int n = 0; n < max_iter; ++n ){
		int next_act_id = -1;

		double prob = d_rand(mt);
		if( mem_id < initial_mem || prob - eps < 0.0 ){ // random action
			std::vector<int> possible;
			for( int i = 0; i < max_id; ++i ) if( func_act(i) ) possible.push_back(i);

			if( possible.size() == 0 ) next_act_id = -100;
			else next_act_id = possible[i_rand(mt)%possible.size()];
		}
		else{
			next_act_id = get_next_action(cur_state.s);
		}

		State next_state = func_trans(next_act_id);
		mem[mem_id%mem_capacity] = DQN_Memory(cur_state.s, next_state.s, next_act_id, next_state.r);
		++mem_id;

		// Is this needed to generate x and d in each loop?
		const int data_size = std::min(mem_id, batch_size);
		const int dim = next_state.s.size();
		std::vector<int> idx(std::min(mem_id, mem_capacity));
		std::vector<std::vector<Vec>> x(data_size), x_n(data_size), d(data_size);

		std::iota( idx.begin(), idx.end(), 0 );
		std::shuffle( idx.begin(), idx.end(), mt );
			
		for( int i = 0; i < data_size; ++i ){
			x[i] = mem[idx[i]].s;
			x_n[i] = mem[idx[i]].s_n;
		}

		auto tmp = Q_tilde->apply(x_n);
		for( int i = 0; i < data_size; ++i ){
			int id = 0;
			double val = tmp[i][0][0];
			for( int j = 1; j < max_id; ++j ){
				if( val < tmp[i][0][j] ){
					id = j;
					val = tmp[i][0][j];
				}
			}
			d[i] = std::vector<Vec>(1, Vec(max_id, 0.0));

			if( next_state.ep_s == DQN_STATE::EPISODE_CONT )
				d[i][0][id] = mem[idx[i]].r + gamma * val;
			else
				d[i][0][id] = mem[idx[i]].r;
		}

		// do backpropagation with mem
		Q->set_BATCHSIZE(data_size);
		Q->learning(x, d, 0);

		cur_state = next_state;
		if( mem_id > initial_mem ){
			if( n % update_model_freq == 0 ){
				// update Q_tilde weights
				for( int i = 0; i < tilde_layers.size(); ++i ){
					*tilde_layers[i] = *layers[i];
				}
			}

			eps -= 1.0/1.0E6;
			if( eps < this->epsilon ) eps = this->epsilon;
		}
	}
}

void DQN::output_W ( const std::string& filename )
{
	Q->output_W(filename);
}

void DQN::set_W ( const std::string& filename )
{
	Q->set_W(filename);
}
