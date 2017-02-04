#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <cmath>
#include <chrono>

#include "../include/Neuralnet.hpp"
#include "../include/Layer/Layer.hpp"
#include "../include/Layer/FullyConnected.hpp"
// #include "../include/Convolutional.hpp"
// #include "../include/Pooling.hpp"

using namespace std;

// pixel normalize function
void normalize ( vector<Matrix<float>>& image, vector<vector<float>>& ave )
{
    for( int i = 0; i < image.size(); ++i ){
        ave.emplace_back(image[i].m, 0.0);

        for( int j = 0; j < image[i].m; ++j ){
            for( int k = 0; k < image[i].n; ++k ) ave[i][j] += image[i](j, k);
        }
        for( int j = 0; j < image[i].m; ++j ) ave[i][j] /= image[i].n;

        for( int j = 0; j < image[i].m; ++j )
			for( int k = 0; k < image[i].n; ++k )
				image[i](j, k) -= ave[i][j];
    }
}

int main( int argc, char* argv[] )
{
	// define mini-batch size.
	const int BATCH_SIZE = 128;

	// construct neuralnetwork with CrossEntropy.
    Neuralnet<clMatrix, float> net(shared_ptr<LossFunction<float>>(new CrossEntropy<float>));
	vector<shared_ptr<Layer<clMatrix, float>>> layers;

	// define layers.
	// layers.emplace_back(new Convolutional(1, 28*28, 28,
	// 									  20, 28*28, 28,
	// 									  5, 5, 1, shared_ptr<Function>(new ReLU)));
	// layers.emplace_back(new Pooling(20, 28*28, 28,
	// 								20, 7*7, 7,
	// 								4, 4, 4, shared_ptr<Function>(new Identity)));	

	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 28*28, 1, 1000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 1000, 1, 500, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 500, 1, 10, shared_ptr<Function<float>>(new Softmax<float>)));

	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 28*28, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 10, shared_ptr<Function<float>>(new Softmax<float>)));

	// this neuralnet has 4 layers, input, convolutional, pooling and FullyConnected.
	for( int i = 0; i < layers.size(); ++i ){
		net.add_layer(layers[i]);
	}
	
	// read a test data of MNIST(http://yann.lecun.com/exdb/mnist/).
	const int N = 10000;
	vector<Matrix<float>> train_x(1, Matrix<float>(28*28, N)),
		train_d(1, Matrix<float>(10, N));
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

	train_image.seekg(4*4, ios_base::beg);
	train_label.seekg(4*2, ios_base::beg);
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

	vector<vector<float>> ave;
	// normalize train image.
	normalize(train_x, ave);

	// read a train data of MNIST.
	const int M = 10000;
	vector<Matrix<float>> test_x(1, Matrix<float>(28*28, M)), test_d(1, Matrix<float>(10, M));
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
		for( int j = 0; j < 10; ++j ) test_d[0](j,i) = 0.0;
		test_d[0](tmp_lab,i) = 1.0;

		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			test_image.read((char*)&c, sizeof(unsigned char));
			test_x[0](j,i) = (c/255.0 - ave[0][j]);
		}
	}
	
	vector<clMatrix<float>> train_X(1), train_D(1);
	vector<clMatrix<float>> test_X(1), test_D(1);

	train_X[0] = train_x[0]; train_D[0] = train_d[0];
	test_X[0] = test_x[0]; test_D[0] = test_d[0];
	// checking error function.
	auto total = chrono::system_clock::now();
	auto prev_time = chrono::system_clock::now();
	auto check_error = [&](const Neuralnet<clMatrix, float>& nn, const int iter, const std::vector<clMatrix<float>>& x, const std::vector<clMatrix<float>>& d ) -> void {
		if( iter%(N/BATCH_SIZE) != 0 ) return;

		auto tmp_time = chrono::system_clock::now();
		long long tmp_cntflop = cnt_flop;

		const int once_num = 1000;
		int train_ans_num = 0;
		for( int i = 0; i < N; i += once_num ){
			int size = min(once_num, N-i);

			vector<clMatrix<float>> tmp_X(train_X.size());
			for( int j = 0; j < train_X.size(); ++j )
				tmp_X[j] = train_X[j].sub(0, i, train_X[j].m, size);

			auto tmp_Y = nn.apply(tmp_X)[0].get_matrix();
			for( int j = 0; j < tmp_Y.n; ++j ){
				int idx = 0, lab;
				double max_num = tmp_Y(0,j);
				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_Y(k,j) ){
						max_num = tmp_Y(k,j);
						idx = k;
					}
					if( train_d[0](k, i+j) == 1.0 ) lab = k;
				}
				if( idx == lab ) ++train_ans_num;
			}
		}

		int test_ans_num = 0;
		for( int i = 0; i < M; i += once_num ){
			int size = min(once_num, N-i);

			vector<clMatrix<float>> tmp_X(test_X.size());
			for( int j = 0; j < test_X.size(); ++j )
				tmp_X[j] = test_X[j].sub(0, i, test_X[j].m, size);

			auto tmp_Y = nn.apply(tmp_X)[0].get_matrix();
			for( int j = 0; j < tmp_Y.n; ++j ){
				int idx = 0, lab;
				double max_num = tmp_Y(0,j);
				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_Y(k,j) ){
						max_num = tmp_Y(k,j);
						idx = k;
					}
					if( test_d[0](k, i+j) == 1.0 ) lab = k;
				}
				if( idx == lab ) ++test_ans_num;
			}
		}

		auto tmp_total = chrono::system_clock::now();

		double t1 = chrono::duration_cast<chrono::nanoseconds>(tmp_time - prev_time).count()/1e9;
		double t2 = chrono::duration_cast<chrono::nanoseconds>(tmp_total - total).count()/1e9;
		
		printf("Iter : %5d, Epoch : %3d\n", iter, iter/(N/BATCH_SIZE));
		printf("  Train data answer rate %3.2f%%\n", (double)train_ans_num/N*100.0);
		printf("  Test data answer rate  %3.2f%%\n", (double)test_ans_num/M*100.0);
		printf("  elapsed time : %.3f[s], total time : %.3f[s]\n", t1, t2);
		printf("  (%.3f, %.3f)[GFLOPS]\n", tmp_cntflop/t1/1e9, cnt_flop/(chrono::duration_cast<chrono::nanoseconds>(tmp_total - prev_time).count()/1e9)/1e9);

		cnt_flop = 0;
		prev_time = chrono::system_clock::now();
	};

	// set a hyper parameter.
	net.set_EPS(1.0E-3);
	net.set_LAMBDA(0.0);
	net.set_BATCHSIZE(BATCH_SIZE);
	// learning the neuralnet in 10 EPOCH and output error defined above in each epoch.
	total = prev_time = chrono::system_clock::now();

	train_X[0] = train_x[0]; train_D[0] = train_d[0];
	net.learning(train_X, train_D, N/BATCH_SIZE*10, check_error);
}
