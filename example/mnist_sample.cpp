#include <iostream>
#include <fstream>
#include <functional>
#include <memory>
#include <cmath>

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

	// construct neuralnetwork with CrossEntropy.
    Neuralnet net(shared_ptr<LossFunction>(new CrossEntropy));
	vector<shared_ptr<Layer>> layers;

	// define layers.
	layers.emplace_back(new Convolutional(1, 28*28, 28,
										  20, 28*28, 28,
										  5, 5, 1, shared_ptr<Function>(new ReLU)));
	layers.emplace_back(new Pooling(20, 28*28, 28,
									20, 7*7, 7,
									4, 4, 4, shared_ptr<Function>(new Identity)));
	layers.emplace_back(new FullyConnected(20, 7*7, 1, 10, shared_ptr<Function>(new Softmax)));

	// this neuralnet has 4 layers, input, convolutional, pooling and FullyConnected.
	for( int i = 0; i < layers.size(); ++i ){
		net.add_layer(layers[i]);
	}
	
	// read a test data of MNIST(http://yann.lecun.com/exdb/mnist/).
	vector<int> train_lab;
	vector<vector<vector<double>>> train_x;
	const int N = 1000;
	ifstream train_image("train-images-idx3-ubyte", ios_base::binary);
	ifstream train_label("train-labels-idx1-ubyte", ios_base::binary);
	train_image.seekg(4*4, ios_base::beg);
	train_label.seekg(4*2, ios_base::beg);
	for( int i = 0; i < N; ++i ){
		unsigned char tmp_lab;
		train_label.read((char*)&tmp_lab, sizeof(unsigned char));
		train_lab.push_back(tmp_lab);
		
		vector<vector<double>> tmp(1, vector<double>(28*28));
		for( int i = 0; i < 28*28; ++i ){
			unsigned char c;
			train_image.read((char*)&c, sizeof(unsigned char));
			tmp[0][i] = (c/255.0);
		}
		
		train_x.push_back(tmp);
	}

	vector<vector<double>> ave;
	// normalize train image.
	normalize(train_x, ave);

	// read a train data of MNIST.
	vector<int> test_lab;
	vector<vector<vector<double>>> test_x;
	const int M = 1000;
	ifstream test_image("t10k-images-idx3-ubyte", ios_base::binary);
	ifstream test_label("t10k-labels-idx1-ubyte", ios_base::binary);
	test_image.seekg(4*4, ios_base::beg);
	test_label.seekg(4*2, ios_base::beg);
	for( int i = 0; i < M; ++i ){
		unsigned char tmp_lab;
		test_label.read((char*)&tmp_lab, sizeof(unsigned char));
		test_lab.push_back(tmp_lab);
		
		vector<vector<double>> tmp(1, vector<double>(28*28));
		for( int i = 0; i < 28*28; ++i ){
			unsigned char c;
			test_image.read((char*)&c, sizeof(unsigned char));
			tmp[0][i] = (c/255.0 - ave[0][i]);
		}
		
		test_x.push_back(tmp);
	}
	
	// checking error function.
	string text = "Train data answer rate : ";
	auto check_error = [&](const Neuralnet& nn, const int iter, const std::vector<Matrix<double>>& x, const std::vector<Matrix<double>>& d ) -> void {
		if( iter%(N/BATCH_SIZE) != 0 || iter == 0 ) return;

		int ans_num = 0;
		auto Y = nn.apply(x);
		for( int i = 0; i < Y[0].n; ++i ){
			int idx, lab;
			double max_num = 0.0;
			for( int j = 0; j < 10; ++j ){
				if( max_num < Y[0](j,i) ){
					max_num = Y[0](j,i);
					idx = j;
				}
				if( d[0](j, i) == 1.0 ) lab = j;
			}
			if( idx == lab ) ++ans_num;
		}

		printf("%s%.2f%%\n", text.c_str(), (double)ans_num/Y[0].n*100.0);
	};

	// set supervised data.
	vector<vector<vector<double>>> d(N, vector<vector<double>>(1, vector<double>(10, 0.0)));
	for( int i = 0; i < N; ++i ) d[i][0][train_lab[i]] = 1.0;

	vector<Matrix<double>> X(1, Matrix<double>(28*28, M)), Y(1, Matrix<double>(10, M));
	for( int i = 0; i < M; ++i ){
		for( int j = 0; j < 28*28; ++j ){
			X[0](j, i) = test_x[i][0][j];
		}
		Y[0](test_lab[i], i) = 1.0;
	}

	// set a hyper parameter.
	net.set_EPS(1.0E-3);
	net.set_LAMBDA(0.0);
	net.set_BATCHSIZE(BATCH_SIZE);
	// learning the neuralnet in 10 EPOCH and output error defined above in each epoch.
	net.learning(train_x, d, N/BATCH_SIZE*10, check_error);

	// calc answer rate of test data.
	text = "Test data answer rate : ";
	check_error(net, N/BATCH_SIZE, X, Y);
}
