#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <cmath>

#include <Neuralnet.hpp>
#include <Layer/Layer.hpp>
#include <Layer/FullyConnected.hpp>
#include <Layer/Convolutional.hpp>

using namespace std;

// pixel normalize function
template<typename T>
void normalize ( int num_map, Matrix<T>& image, vector<vector<double>>& ave )
{
	int leng = image.m / num_map;
    for( int i = 0; i < num_map; ++i ){
        ave.emplace_back(leng, 0.0);

        for( int j = 0; j < leng; ++j ){
            for( int k = 0; k < image.n; ++k ) ave[i][j] += image(i*leng + j,k);
        }
        for( int j = 0; j < leng; ++j ) ave[i][j] /= image.n;

        for( int j = 0; j < leng; ++j )
			for( int k = 0; k < image.n; ++k )
				image(leng*i + j,k) -= ave[i][j];
    }
}

int main( int argc, char* argv[] )
{
	// define mini-batch size.
	const int BATCH_SIZE = 128;
	typedef double Real;
	
	// construct neuralnetwork with CrossEntropy.
    Neuralnet<Matrix, Real> net(shared_ptr<LossFunction<Real>>(new CrossEntropy<Real>));
	vector<shared_ptr<Layer<Matrix, Real>>> layers;

	// define layers.
	layers.emplace_back(new Convolutional<Matrix, Real>(1, 28*28, 28,
														32, 28*28, 28,
														3, 3, 1, 1, shared_ptr<Function<Real>>(new ReLU<Real>)));
	layers.emplace_back(new Convolutional<Matrix, Real>(32, 28*28, 28,
														64, 14*14, 14,
														3, 3, 2, 1, shared_ptr<Function<Real>>(new ReLU<Real>)));
	layers.emplace_back(new FullyConnected<Matrix, Real>(64, 14*14, 1, 1000, shared_ptr<Function<Real>>(new ReLU<Real>)));
	layers.emplace_back(new FullyConnected<Matrix, Real>(1, 1000, 1, 500, shared_ptr<Function<Real>>(new ReLU<Real>)));
	layers.emplace_back(new FullyConnected<Matrix, Real>(1, 500, 1, 10, shared_ptr<Function<Real>>(new Softmax<Real>)));

	// this neuralnet has 4 layers, input, convolutional, pooling and FullyConnected.
	for( unsigned int i = 0; i < layers.size(); ++i ){
		net.add_layer(layers[i]);
	}
	
	// read a train data of MNIST(http://yann.lecun.com/exdb/mnist/).
	const int N = 10000;
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
	Matrix<Real> train_x(28*28, N), train_d(10, N);
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

	vector<vector<double>> ave;
	// normalize train image.
	normalize(1, train_x, ave);

	// read a test data of MNIST.
	const int M = 5000;
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
	Matrix<Real> test_x(28*28, M), test_d(10, M);
	for( int i = 0; i < M; ++i ){
		unsigned char tmp_lab;
		test_label.read((char*)&tmp_lab, sizeof(unsigned char));
		for( int j = 0; j < 10; ++j ) test_d(j, i) = 0.0;
		test_d(tmp_lab, i) = 1.0;
		
		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			test_image.read((char*)&c, sizeof(unsigned char));
			test_x(j,i) = (c/255.0 - ave[0][j]);
		}
	}
	
	chrono::time_point<chrono::system_clock> prev_time, total_time;
	// checking error function.
	auto check_error = [&](const Neuralnet<Matrix, Real>& nn, const int iter, const Matrix<Real>& x, const Matrix<Real>& d ) -> void {
		if( iter%(N/BATCH_SIZE) != 0 ) return;

		// extracting number of samples from data(for reduciong memory consumption)
		const int once_num = 1000;

		auto tmp_time = chrono::system_clock::now();
		double flops = (double)cnt_flop / (std::chrono::duration_cast<std::chrono::milliseconds>(tmp_time - prev_time).count()/1e3) / 1e9;
		
		// calculate answer rate of train data
		int train_ans_num = 0;
		for( int i = 0; i < N; i += once_num ){
			int size = min(once_num, N - i);
			Matrix<Real> tmp_x = train_x.sub(0, i, 28*28, size);
			
			auto tmp_y = nn.apply(tmp_x);
			for( int j = 0; j < tmp_y.n; ++j ){
				int idx = 0, lab = -1;
				double max_num = tmp_y(0, j);

				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_y(k, j) ){
						max_num = tmp_y(k, j);
						idx = k;
					}
					if( train_d(k, i+j) == 1.0 ) lab = k;
				}
				if( idx == lab ) ++train_ans_num;
			}
		}

		// calculate answer rate of test data
		int test_ans_num = 0;
		for( int i = 0; i < M; i += once_num ){
			int size = min(once_num, M - i);
			Matrix<Real> tmp_x = test_x.sub(0, i, 28*28, size);
			
			auto tmp_y = nn.apply(tmp_x);
			for( int j = 0; j < tmp_y.n; ++j ){
				int idx = 0, lab = -1;
				double max_num = tmp_y(0, j);

				for( int k = 0; k < 10; ++k ){
					if( max_num < tmp_y(k, j) ){
						max_num = tmp_y(k, j);
						idx = k;
					}
					if( test_d(k, i+j) == 1.0 ) lab = k;
				}
				if( idx == lab ) ++test_ans_num;
			}
		}

		printf("Iter %5d, Epoch %3d\n", iter, iter/(N/BATCH_SIZE));
		printf("  Elapsed time : %.3f, Total time : %.3f\n",
			   chrono::duration_cast<chrono::milliseconds>(tmp_time - prev_time).count()/1e3,
			   chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now() - total_time).count()/1e3);
		printf("  Train answer rate %.2f%%\n", (double)train_ans_num/N*100.0);
		printf("  Test answer rate  %.2f%%\n", (double)test_ans_num/M*100.0);
		printf("  %.3f[GFLOPS]\n\n", flops);
		prev_time = chrono::system_clock::now();
		cnt_flop = 0;
	};

	// set supervised data.

	// set a hyper parameter.
	net.set_EPS(1.0E-3);
	net.set_LAMBDA(0.0);
	net.set_BATCHSIZE(BATCH_SIZE);
	// learning the neuralnet in 10 EPOCH and output error defined above in each epoch.
	prev_time = total_time = chrono::system_clock::now();
	net.learning(train_x, train_d, N/BATCH_SIZE*10, check_error);
}
