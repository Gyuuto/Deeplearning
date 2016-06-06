#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <cmath>
#include <cstring>

#include "../include/Neuralnet.hpp"
#include "../include/Layer.hpp"
#include "../include/FullyConnected.hpp"
#include "../include/Convolutional.hpp"
#include "../include/Pooling.hpp"

using namespace std;

// pixel normalize function
void normalize ( vector<vector<vector<double>>>& image, vector<vector<double>>& ave )
{
    for( int i = 0; i < image[0].size(); ++i ){
        ave.emplace_back(image[0][0].size(), 0.0);

        for( int j = 0; j < image.size(); ++j ){
            for( int k = 0; k < image[0][i].size(); ++k ) ave[i][k] += image[j][i][k];
        }
        for( int j = 0; j < image[0][i].size(); ++j ) ave[i][j] /= image.size();

        for( int j = 0; j < image.size(); ++j )
			for( int k = 0; k < image[0][i].size(); ++k )
				image[j][i][k] -= ave[i][k];
    }
}

int main( int argc, char* argv[] )
{
	// define mini-batch size.
	const int BATCH_SIZE = 50;

	// parallelism for model parallel.
	int INNER = 1;
	for( int i = 1; i < argc; ++i ){
		if( strcmp(argv[i], "--inner") == 0 ){
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
    Neuralnet net(shared_ptr<LossFunction>(new CrossEntropy), outer_world, inner_world);
	vector<shared_ptr<Layer>> layers;

	// define layers.
	layers.emplace_back(new Convolutional(1, 28*28, 28,
										  10, 28*28, 28,
										  5, 5, 1, shared_ptr<Function>(new ReLU)));
	layers.emplace_back(new Pooling(10, 28*28, 28,
									10, 14*14, 14,
									2, 2, 2, shared_ptr<Function>(new Identity)));
	layers.emplace_back(new Convolutional(10, 14*14, 14,
										  20, 14*14, 14,
										  5, 5, 1, shared_ptr<Function>(new ReLU)));
	layers.emplace_back(new Pooling(20, 14*14, 14,
									20, 7*7, 7,
									2, 2, 2, shared_ptr<Function>(new Identity)));
	layers.emplace_back(new FullyConnected(20, 7*7, 1, 512, shared_ptr<Function>(new ReLU)));
	layers.emplace_back(new FullyConnected(1, 512, 1, 10, shared_ptr<Function>(new Softmax)));

	// this neuralnet has 7 layers, input, Conv1, MaxPool, Conv2, MaxPool, Fc1 and Fc2;
	for( int i = 0; i < layers.size(); ++i ){
		net.add_layer(layers[i]);
	}
	
	// read a test data of MNIST(http://yann.lecun.com/exdb/mnist/).
	vector<int> train_lab;
	vector<vector<vector<double>>> train_x;
	const int N = 60000 / outer_nprocs;
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

	train_image.seekg(4*4 + 28*28*outer_rank * N, ios_base::beg);
	train_label.seekg(4*2 + outer_rank * N, ios_base::beg);
	for( int i = 0; i < N; ++i ){
		unsigned char tmp_lab;
		train_label.read((char*)&tmp_lab, sizeof(unsigned char));
		train_lab.push_back(tmp_lab);
		
		vector<vector<double>> tmp(1, vector<double>(28*28));
		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			train_image.read((char*)&c, sizeof(unsigned char));
			tmp[0][j] = (c/255.0);
		}
		
		train_x.push_back(tmp);
	}
	
	// set supervised data.
	vector<vector<vector<double>>> d(N, vector<vector<double>>(1, vector<double>(10, 0.0)));
	for( int i = 0; i < N; ++i ) d[i][0][train_lab[i]] = 1.0;

	vector<vector<double>> ave;
	// normalize train image.
	normalize(train_x, ave);

	// read a train data of MNIST.
	vector<int> test_lab;
	vector<vector<vector<double>>> test_x;
	const int M = 10000;
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
	for( int i = 0; i < M; ++i ){
		unsigned char tmp_lab;
		test_label.read((char*)&tmp_lab, sizeof(unsigned char));
		test_lab.push_back(tmp_lab);
		
		vector<vector<double>> tmp(1, vector<double>(28*28));
		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			test_image.read((char*)&c, sizeof(unsigned char));
			tmp[0][j] = (c/255.0 - ave[0][j]);
		}
		
		test_x.push_back(tmp);
	}

	// train data into Matrix class for checking error.
	vector<Matrix<double>> X(1, Matrix<double>(28*28, M)), D(1, Matrix<double>(10, M));
	for( int i = 0; i < M; ++i ){
		for( int j = 0; j < 28*28; ++j ){
			X[0](j, i) = test_x[i][0][j];
		}
		D[0](test_lab[i], i) = 1.0;
	}

	// checking error function.
	double prev_time, total_time;
	auto check_error = [&](const Neuralnet& nn, const int iter, const std::vector<Matrix<double>>& x, const std::vector<Matrix<double>>& d ) -> void {
		if( iter%((N/BATCH_SIZE)) != 0 || iter == 0 ) return;

		const int once_num = 1000;
		double tmp_time = MPI_Wtime();

		// calculating answer rate among all train data.
		int train_ans_num = 0;
		for( int i = 0; i < train_x.size(); i += once_num ){
			int size = min(once_num, (int)train_x.size() - i);
			vector<vector<vector<double>>> tmp_x(size);
			for( int j = 0; j < size; ++j ) tmp_x[j] = train_x[i+j];

			auto y = nn.apply(tmp_x);
			for( int j = 0; j < y.size(); ++j ){
				int idx, lab;
				double max_num = 0.0;
				for( int k = 0; k < 10; ++k ){
					if( max_num < y[j][0][k] ){
						max_num = y[j][0][k];
						idx = k;
					}
				}
				if( idx == train_lab[i+j] ) ++train_ans_num;
			}
		}

		// calculating answer rate among all test data.
		int test_ans_num = 0;
		for( int i = 0; i < X[0].n; i += once_num ){
			int size = min(once_num, X[0].n - i);
			vector<Matrix<double>> tmp_X(X.size(), Matrix<double>(X[0].m, size));
			for( int j = 0; j < size; ++j )
				for( int k = 0; k < X[0].m; ++k )
					tmp_X[0](k, j) = X[0](k, i+j);

			auto Y = nn.apply(tmp_X);
			for( int j = 0; j < Y[0].n; ++j ){
				int idx, lab;
				double max_num = 0.0;
				for( int k = 0; k < 10; ++k ){
					if( max_num < Y[0](k,j) ){
						max_num = Y[0](k,j);
						idx = k;
					}
				}
				if( idx == test_lab[i+j] ) ++test_ans_num;
			}
		}

		// output answer rate each test data batch.
		if( inner_rank == 0 ){
			for( int i = 0; i < outer_nprocs; ++i ){
				if( i == outer_rank ){
					printf("outer rank : %d\n", outer_rank);
					printf("  Elapsed time : %.3f, Total time : %.3f\n", tmp_time - prev_time, MPI_Wtime() - total_time);
					printf("  Train answer rate : %.2f%%\n", (double)train_ans_num/train_x.size()*100.0);
					printf("  Test answer rate  : %.2f%%\n", (double)test_ans_num/X[0].n*100.0);
				}
				MPI_Barrier(outer_world);
			}
			prev_time = MPI_Wtime();
		}
		if( world_rank == 0 ) puts("");
	};

	// set a hyper parameter.
	net.set_EPS(1.0E-3);
	net.set_LAMBDA(0.0);
	net.set_BATCHSIZE(BATCH_SIZE);
	net.set_UPDATEITER(N/BATCH_SIZE*2);
	// learning the neuralnet in 20 EPOCH and output error defined above in each epoch.
	prev_time = total_time = MPI_Wtime();
	net.learning(train_x, d, N/BATCH_SIZE*10, check_error);

	MPI_Finalize();
}
