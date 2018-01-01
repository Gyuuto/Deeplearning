#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <cmath>
#include <chrono>

#include <Neuralnet.hpp>
#include <Layer/Layer.hpp>
#include <Layer/FullyConnected.hpp>
#include <Layer/Convolutional.hpp>
// #include "../include/Pooling.hpp"

using namespace std;

// pixel normalize function
void normalize ( int num_map, Matrix<float>& image, vector<vector<float>>& ave )
{
	int leng = image.m / num_map;
    for( int i = 0; i < num_map; ++i ){
        ave.emplace_back(leng, 0.0);

        for( int j = 0; j < leng; ++j ){
            for( int k = 0; k < image.n; ++k ) ave[i][j] += image(i*leng + j, k);
        }
        for( int j = 0; j < leng; ++j ) ave[i][j] /= image.n;

        for( int j = 0; j < leng; ++j )
			for( int k = 0; k < image.n; ++k )
				image(leng*i + j, k) -= ave[i][j];
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
	layers.emplace_back(new Convolutional<clMatrix, float>(1, 28*28, 28,
														   32, 28*28, 28,
														   3, 3, 1, 1, shared_ptr<Function<float>>(new ReLU<float>)));
	((Convolutional<clMatrix, float>*)(layers.rbegin()->get()))->set_once_num(BATCH_SIZE);
	layers.emplace_back(new Convolutional<clMatrix, float>(32, 28*28, 28,
														   64, 14*14, 14,
														   3, 3, 2, 1, shared_ptr<Function<float>>(new ReLU<float>)));
	((Convolutional<clMatrix, float>*)(layers.rbegin()->get()))->set_once_num(BATCH_SIZE);

	layers.emplace_back(new FullyConnected<clMatrix, float>(64, 14*14, 1, 1000, shared_ptr<Function<float>>(new ReLU<float>)));
	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 1000, 1, 500, shared_ptr<Function<float>>(new ReLU<float>)));
	layers.emplace_back(new FullyConnected<clMatrix, float>(1, 500, 1, 10, shared_ptr<Function<float>>(new Softmax<float>)));

	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 28*28, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 2000, shared_ptr<Function<float>>(new ReLU<float>)));
	// layers.emplace_back(new FullyConnected<clMatrix, float>(1, 2000, 1, 10, shared_ptr<Function<float>>(new Softmax<float>)));

	// this neuralnet has 4 layers, input, convolutional, pooling and FullyConnected.
	for( unsigned int i = 0; i < layers.size(); ++i ){
		net.add_layer(layers[i]);
	}
	
	// read a train data of MNIST(http://yann.lecun.com/exdb/mnist/).
	const int N = 10000;
	Matrix<float> train_x(28*28, N), train_d(10, N);
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
		for( int j = 0; j < 10; ++j ) train_d(j, i) = 0.0;
		train_d(tmp_lab, i) = 1.0;
		
		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			train_image.read((char*)&c, sizeof(unsigned char));
			train_x(j, i) = (c/255.0);
		}
	}

	vector<vector<float>> ave;
	// normalize train image.
	normalize(1, train_x, ave);

	// read a test data of MNIST.
	const int M = 5000;
	Matrix<float> test_x(28*28, M), test_d(10, M);
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
		for( int j = 0; j < 10; ++j ) test_d(j,i) = 0.0;
		test_d(tmp_lab,i) = 1.0;

		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			test_image.read((char*)&c, sizeof(unsigned char));
			test_x(j,i) = (c/255.0 - ave[0][j]);
		}
	}
	
	clMatrix<float> train_X = train_x, train_D = train_d;
	clMatrix<float> test_X = test_x, test_D = test_d;

	// checking error function.
	auto total = chrono::system_clock::now();
	auto prev_time = chrono::system_clock::now();
	auto check_error = [&](const Neuralnet<clMatrix, float>& nn, const int iter, const clMatrix<float>& x, const clMatrix<float>& d ) -> void {
		if( iter%(N/BATCH_SIZE) != 0 ) return;

		auto tmp_time = chrono::system_clock::now();
		long long tmp_cntflop = cnt_flop;

		const int once_num = 1000;
		int train_ans_num = 0;
		for( int i = 0; i < N; i += once_num ){
			int size = min(once_num, N-i);

			clMatrix<float> tmp_X = train_X.sub(0, i, train_X.m, size);

			auto tmp_Y = nn.apply(tmp_X).get_matrix();
			for( int j = 0; j < tmp_Y.n; ++j ){
				int idx = 0, lab = -1;
				double max_num = tmp_Y(0,j);
				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_Y(k,j) ){
						max_num = tmp_Y(k,j);
						idx = k;
					}
					if( train_d(k, i+j) == 1.0 ) lab = k;
				}
				if( idx == lab ) ++train_ans_num;
			}
		}

		int test_ans_num = 0;
		for( int i = 0; i < M; i += once_num ){
			int size = min(once_num, N-i);

			clMatrix<float> tmp_X = test_X.sub(0, i, test_X.m, size);

			auto tmp_Y = nn.apply(tmp_X).get_matrix();
			for( int j = 0; j < tmp_Y.n; ++j ){
				int idx = 0, lab = -1;
				double max_num = tmp_Y(0,j);
				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_Y(k,j) ){
						max_num = tmp_Y(k,j);
						idx = k;
					}
					if( test_d(k, i+j) == 1.0 ) lab = k;
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

	net.learning(train_X, train_D, N/BATCH_SIZE*20, check_error);
}
