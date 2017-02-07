#include <iostream>
#include <functional>
#include <memory>
#include <cmath>

#include "../include/Neuralnet.hpp"
#include "../include/Layer.hpp"
#include "../include/FullyConnected.hpp"
#include "../include/Function.hpp"

using namespace std;

int main()
{
	Neuralnet net(shared_ptr<LossFunction>(new Square));
	vector<shared_ptr<Layer>> layers;
	// define layers
	layers.emplace_back(new FullyConnected(1, 1, 1, 100, shared_ptr<Function>(new ReLU)));
	layers.emplace_back(new FullyConnected(1, 100, 1, 1, shared_ptr<Function>(new Identity)));

	// this neuralnet has 3 layer, input, hidden and output.
	net.add_layer(layers[0]);
	net.add_layer(layers[1]);

	// set input values and supervised values.
	vector<vector<vector<double>>> x(100), y(100);
	for( int i = 0; i < 100; ++i ){
		x[i] = {{-1.0 + 2.0/99.0*i}};
		y[i] = {{cos(x[i][0][0])}};
	}
	
	// set a hyper parameter.
	net.set_EPS(1.0E-3);
	net.set_LAMBDA(0.0);
	net.set_BATCHSIZE(10);
	// learning the neuralnet in 10 EPOCH.
	net.learning(x, y, 100/10*10);
	
	// check approximated function by neuralnet.
	vector<vector<vector<double>>> input = {{{0.8}}};
	auto output = net.apply(input);
	printf("Approximation : %.6E\n", output[0][0][0]);
	printf("Correct       : %.6E\n", cos(input[0][0][0]));
}
