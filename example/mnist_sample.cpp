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

int main()
{
	// define some function and differential.
	auto ReLU = [](double x) -> double { return max(0.0, x); };
	auto dReLU = [](double x) -> double { return (x <= 0.0 ? 0.0 : 1.0); };
	auto idf = [](double x) -> double { return x; };
	auto didf = [](double x) -> double { return 1.0; };

    Neuralnet net;
	vector<shared_ptr<Layer>> layers;
	// define layers.
	layers.emplace_back(new Convolutional(1, 28*28, 28,
										  20, 28*28, 28,
										  5, 5, 1, ReLU, dReLU));
	layers.emplace_back(new Pooling(20, 28*28, 28,
										  20, 7*7, 7,
										  4, 4, 4, idf, didf));
	layers.emplace_back(new FullyConnected(20, 7*7, 1, 10, idf, didf));

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
	ifstream test_image("test-images-idx3-ubyte", ios_base::binary);
	ifstream test_label("test-labels-idx1-ubyte", ios_base::binary);
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
	auto check_error = [&](const Neuralnet& nn) -> void {
		int ans_num = 0;
		auto Y = nn.apply(train_x);
		for( int i = 0; i < N; ++i ){
			vector<double> prob(10, 0.0);
			int idx;
			double sum = 0.0, max_num = 0.0;
			for( int j = 0; j < 10; ++j ) sum += exp(Y[i][0][j]);
			for( int j = 0; j < 10; ++j ){
				prob[j] = exp(Y[i][0][j]) / sum;
				if( max_num < prob[j] ){
					max_num = prob[j];
					idx = j;
				}
			}
			if( idx == train_lab[i] ) ++ans_num;
		}
		printf("Train data answer rate : %.2f%%\n", (double)ans_num/N*100.0);
	
		ans_num = 0;
		Y = nn.apply(test_x);
		for( int i = 0; i < M; ++i ){
			vector<double> prob(10, 0.0);
			int idx;
			double sum = 0.0, max_num = 0.0;
			for( int j = 0; j < 10; ++j ) sum += exp(Y[i][0][j]);
			for( int j = 0; j < 10; ++j ){
				prob[j] = exp(Y[i][0][j]) / sum;
				if( max_num < prob[j] ){
					max_num = prob[j];
					idx = j;
				}
			}
			if( idx == test_lab[i] ) ++ans_num;
		}
		printf("Test   data answer rate : %.2f%%\n", (double)ans_num/N*100.0);
	};
	
	// set supervised data.
	vector<vector<vector<double>>> d(N, vector<vector<double>>(1, vector<double>(10, 0.0)));
	for( int i = 0; i < N; ++i ) d[i][0][train_lab[i]] = 1.0;

	// learning the neuralnet in 10 EPOCH and output error defined above in each epoch.
    net.learning(train_x, d, N*10, check_error);
}
