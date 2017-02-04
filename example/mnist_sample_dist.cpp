#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <cmath>
#include <cstring>

#include <chrono>

#include "../include/Neuralnet.hpp"
#include "../include/Layer/Layer.hpp"
#include "../include/Layer/FullyConnected.hpp"

using namespace std;

// pixel normalize function
void normalize ( vector<Matrix<double>>& image, vector<vector<double>>& ave )
{
    for( int i = 0; i < image.size(); ++i ){
        ave.emplace_back(image[i].m, 0.0);

        for( int j = 0; j < image[i].m; ++j ){
            for( int k = 0; k < image[i].n; ++k ) ave[i][j] += image[i](j,k);
        }
        for( int j = 0; j < image[i].m; ++j ) ave[i][j] /= image[i].n;

        for( int j = 0; j < image[i].m; ++j )
			for( int k = 0; k < image[i].n; ++k )
				image[i](j,k) -= ave[i][j];
    }
}

int main( int argc, char* argv[] )
{
	// define mini-batch size.
	int INNER = 1;
	int N = 60000, M = 10000;
	int BATCH_SIZE = 32, NUM_EPOCH = 20, UPDATE_EPOCH = 1;
	double EPS = 1.0E-3, LAMBDA = 0.0;
	
	for( int i = 1; i < argc; ++i ){
		if( i+1 < argc && strcmp(argv[i], "--N") == 0 ){
			sscanf(argv[i+1], "%d", &N);
			++i;
		}
		else if( i+1 < argc && strcmp(argv[i], "--M") == 0 ){
			sscanf(argv[i+1], "%d", &M);
			++i;
		}
		else if( i+1 < argc && strcmp(argv[i], "--batch") == 0 ){
			sscanf(argv[i+1], "%d", &BATCH_SIZE);
			++i;
		}
		else if( i+1 < argc && strcmp(argv[i], "--num_epoch") == 0 ){
			sscanf(argv[i+1], "%d", &NUM_EPOCH);
			++i;
		}
		else if( i+1 < argc && strcmp(argv[i], "--update_epoch") == 0 ){
			sscanf(argv[i+1], "%d", &UPDATE_EPOCH);
			++i;
		}
		else if( i+1 < argc && strcmp(argv[i], "--eps") == 0 ){
			sscanf(argv[i+1], "%lf", &EPS);
			++i;
		}
		else if( i+1 < argc && strcmp(argv[i], "--lambda") == 0 ){
			sscanf(argv[i+1], "%lf", &LAMBDA);
			++i;
		}
		else if( i+1 < argc && strcmp(argv[i], "--inner") == 0 ){
			sscanf(argv[i+1], "%d", &INNER);
			++i;
		}
	}

	MPI_Init(&argc, &argv);

	MPI_Comm outer_world, inner_world;
	int world_rank, world_nprocs;       // rank and number of processes in all processes.
	int inner_rank, inner_nprocs;       // rank and number of processes in model parallel.
	int outer_rank, outer_nprocs;       // rank and number of processes in data parallel.

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_nprocs);

	// determine model parallel's rank of each processes.
	MPI_Comm_split(MPI_COMM_WORLD, world_rank / INNER, world_rank, &inner_world);
	MPI_Comm_rank(inner_world, &inner_rank);
	MPI_Comm_size(inner_world, &inner_nprocs);

	// determine data parallel's rank of each processes.
	MPI_Comm_split(MPI_COMM_WORLD, world_rank % INNER, world_rank, &outer_world);
	MPI_Comm_rank(outer_world, &outer_rank);
	MPI_Comm_size(outer_world, &outer_nprocs);

	// construct neuralnetwork with CrossEntropy.
    Neuralnet<Matrix, double> net(shared_ptr<LossFunction<double>>(new CrossEntropy<double>), outer_world, inner_world);
	vector<shared_ptr<Layer<Matrix, double>>> layers;

	// define layers.
	layers.emplace_back(new FullyConnected<Matrix, double>(1, 28*28, 1, 1000, shared_ptr<Function<double>>(new ReLU<double>)));
	layers.emplace_back(new FullyConnected<Matrix, double>(1, 1000, 1, 500, shared_ptr<Function<double>>(new ReLU<double>)));
	layers.emplace_back(new FullyConnected<Matrix, double>(1, 500, 1, 10, shared_ptr<Function<double>>(new Softmax<double>)));

	// this neuralnet has 6 layers, Conv1, MaxPool, Conv2, MaxPool, Fc1 and Fc2;
	for( int i = 0; i < layers.size(); ++i ){
		net.add_layer(layers[i]);
	}
	
	// read a test data of MNIST(http://yann.lecun.com/exdb/mnist/).
	ifstream train_image("train-images-idx3-ubyte", ios_base::binary);
	if( !train_image.is_open() ){
		cerr << "\"train-images-idx3-ubyte\" is not found!" << endl;
		return 1;
	}
	ifstream train_label("train-labels-idx1-ubyte", ios_base::binary);
	if( !train_label.is_open() ){
		cerr << "\"train-labels-idx1-ubyte\" is not found!" << endl;
		return 1;
	}

	int org_N = N;
	N = N / outer_nprocs;
	train_image.seekg(4*4 + 28*28*outer_rank * N, ios_base::beg);
	train_label.seekg(4*2 + outer_rank * N, ios_base::beg);
	vector<Matrix<double>> train_x(1, Matrix<double>(28*28, N)), train_d(1, Matrix<double>(10, N));
	for( int i = 0; i < N; ++i ){
		unsigned char tmp_lab;
		train_label.read((char*)&tmp_lab, sizeof(unsigned char));
		for( int j = 0; j < 10; ++j ) train_d[0](j, i) = 0.0;
		train_d[0](tmp_lab, i) = 1.0;
		
		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			train_image.read((char*)&c, sizeof(unsigned char));
			train_x[0](j, i) = (c/255.0);
		}
	}
	
	vector<vector<double>> ave;
	// normalize train image.
	normalize(train_x, ave);

	for( int i = 0; i < ave[0].size(); ++i ) ave[0][i] *= N;
	MPI_Allreduce(MPI_IN_PLACE, &ave[0][0], ave[0].size(), MPI_DOUBLE_PRECISION, MPI_SUM, outer_world);
	for( int i = 0; i < ave[0].size(); ++i ) ave[0][i] /= org_N;

	// read a train data of MNIST.
	ifstream test_image("t10k-images-idx3-ubyte", ios_base::binary);
	if( !test_image.is_open() ){
		cerr << "\"t10k-images-idx3-ubyte\" is not found!" << endl;
		return 1;
	}
	ifstream test_label("t10k-labels-idx1-ubyte", ios_base::binary);
	if( !test_label.is_open() ){
		cerr << "\"t10k-labels-idx1-ubyte\" is not found!" << endl;
		return 1;
	}

	test_image.seekg(4*4, ios_base::beg);
	test_label.seekg(4*2, ios_base::beg);
	vector<Matrix<double>> test_x(1, Matrix<double>(28*28, M)), test_d(1, Matrix<double>(10, M));
	for( int i = 0; i < M; ++i ){
		unsigned char tmp_lab;
		test_label.read((char*)&tmp_lab, sizeof(unsigned char));
		for( int j = 0; j < 10; ++j ) test_d[0](j, i) = 0.0;
		test_d[0](tmp_lab, i) = 1.0;
		
		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			test_image.read((char*)&c, sizeof(unsigned char));
			test_x[0](j, i) = (c/255.0);
		}
	}

	// checking error function.
	chrono::time_point<chrono::system_clock> prev_time, total_time;
	auto check_error = [&](const Neuralnet<Matrix, double>& nn, const int iter, const std::vector<Matrix<double>>& x, const std::vector<Matrix<double>>& d ) -> void {
		if( iter%(N/BATCH_SIZE) != 0 ) return;

		const int once_num = 1000;

		auto tmp_time = chrono::system_clock::now();
		MPI_Allreduce(MPI_IN_PLACE, &cnt_flop, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
		double flops = (double)cnt_flop / (std::chrono::duration_cast<std::chrono::milliseconds>(tmp_time - prev_time).count()/1e3) / 1e9;
		
		// calculating answer rate among all train data.
		int train_ans_num = 0;
		for( int i = 0; i < N; i += once_num ){
			int size = min(once_num, N - i);

			vector<Matrix<double>> tmp_x(train_x.size());
			for( int j = 0; j < train_x.size(); ++j ) tmp_x[j] = train_x[j].sub(0, i, 28*28, size);

			auto tmp_y = nn.apply(tmp_x);
			for( int j = 0; j < size; ++j ){
				int idx = 0, lab = -1;

				double max_num = tmp_y[0](0, j);
				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_y[0](k, j) ){
						max_num = tmp_y[0](k, j);
						idx = k;
					}
					if( train_d[0](k, i+j) == 1.0 ) lab = k;
				}
				if( idx == lab ) ++train_ans_num;
			}
		}

		// calculating answer rate among all test data.
		int test_ans_num = 0;
		for( int i = 0; i < M; i += once_num ){
			int size = min(once_num, M - i);

			vector<Matrix<double>> tmp_x(test_x.size());
			for( int j = 0; j < tmp_x.size(); ++j ) tmp_x[j] = test_x[j].sub(0, i, 28*28, size);

			auto tmp_y = nn.apply(tmp_x);
			for( int j = 0; j < size; ++j ){
				int idx = 0, lab = -1;
				
				double max_num = tmp_y[0](0,j);
				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_y[0](k,j) ){
						max_num = tmp_y[0](k,j);
						idx = k;
					}
					if( test_d[0](k, i+j) == 1.0 ) lab = k;
				}
				if( idx == lab ) ++test_ans_num;
			}
		}

		// output answer rate each test data batch.
		if( inner_rank == 0 ){
			for( int i = 0; i < outer_nprocs; ++i ){
				if( i == outer_rank ){
					printf("outer rank : %3d, Iter : %5d, Epoch %3d\n", outer_rank, iter, iter/(N/BATCH_SIZE));
					printf("  Elapsed time : %.3f, Total time : %.3f\n",
						   chrono::duration_cast<chrono::milliseconds>(tmp_time - prev_time).count()/1e3,
						   chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - total_time).count()/1e3);
					printf("  Train answer rate : %.2f%%\n", (double)train_ans_num/N*100.0);
					printf("  Test answer rate  : %.2f%%\n", (double)test_ans_num/M*100.0);
				}
				MPI_Barrier(outer_world);
			}
			prev_time = chrono::system_clock::now();
		}
		if( world_rank == 0 )
			printf("  %.3f[GFLOPS]\n\n", flops);
		cnt_flop = 0;
	};

	// set a hyper parameter.
	net.set_EPS(EPS);
	net.set_LAMBDA(LAMBDA);
	net.set_BATCHSIZE(BATCH_SIZE);
	net.set_UPDATEITER(N/BATCH_SIZE*UPDATE_EPOCH);
	// learning the neuralnet in SOME EPOCH and output error defined above in each epoch.
	prev_time = total_time = chrono::system_clock::now();
	net.learning(train_x, train_d, N/BATCH_SIZE*NUM_EPOCH, check_error);

	if( world_rank == 0 ){
		for( int i = 0; i < layers.size(); ++i ){
			printf("Layer : %d\n", i);
			printf("    Apply %8.3f[s], init %8.3f[s], gemm %8.3f[s], replacement %8.3f[s]\n",
				   layers[i]->t_apply, layers[i]->t_apply_init, layers[i]->t_apply_gemm, layers[i]->t_apply_repl);
			printf("    Delta %8.3f[s], init %8.3f[s], gemm %8.3f[s], replacement %8.3f[s]\n",
				   layers[i]->t_delta, layers[i]->t_delta_init, layers[i]->t_delta_gemm, layers[i]->t_delta_repl);
			printf("    Grad  %8.3f[s], init %8.3f[s], gemm %8.3f[s], replacement %8.3f[s]\n",
				   layers[i]->t_grad, layers[i]->t_grad_init, layers[i]->t_grad_gemm, layers[i]->t_grad_repl);
		}
	}
	
	MPI_Finalize();
}
