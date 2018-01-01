#include <iostream>
#include <functional>
#include <memory>
#include <cmath>

#include <Function.hpp>
#include <Matrix.hpp>
#include <Neuralnet.hpp>
#include <Layer/Layer.hpp>
#include <Layer/FullyConnected.hpp>

using namespace std;

int main()
{
	Neuralnet<Matrix, double> net(shared_ptr<LossFunction<double>>(new Square<double>));
	vector<shared_ptr<Layer<Matrix, double>>> layers;
	// define layers
	layers.emplace_back(new FullyConnected<Matrix, double>(1, 1, 1, 100, shared_ptr<Function<double>>(new ReLU<double>)));
	layers.emplace_back(new FullyConnected<Matrix, double>(1, 100, 1, 1, shared_ptr<Function<double>>(new Identity<double>)));

	// this neuralnet has 3 layer, input, hidden and output.
	net.add_layer(layers[0]);
	net.add_layer(layers[1]);

	// set input values and supervised values.
    Matrix<double> x(1, 100), y(1, 100);
	for( int i = 0; i < 100; ++i ){
		x(0, i) = -1.0 + 2.0/99.0*i;
		y(0, i) = cos(x(0,i));
	}
	
	// set a hyper parameter.
	net.set_EPS(1.0E-3);
	net.set_LAMBDA(0.0);
	net.set_BATCHSIZE(10);
	// learning the neuralnet in 10 EPOCH.
	net.learning(x, y, 100/10*10);
	
	// check approximated function by neuralnet.
    Matrix<double> input(1, 1); input(0,0) = 0.8;
	auto output = net.apply(input);
	printf("Approximation : %.6E\n", output(0,0));
	printf("Correct       : %.6E\n", cos(input(0,0)));
}
