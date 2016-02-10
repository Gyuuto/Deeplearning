#include <iostream>
#include <memory>
#include <cmath>

#include "Neuralnet.hpp"
#include "Layer.hpp"
#include "FullyConnected.hpp"

using namespace std;

int main()
{
	auto ReLU = [](double x) -> double { return max(0.0, x); };
	auto dReLU = [](double x) -> double { return (x <= 0.0 ? 0.0 : 1.0); };
	auto idf = [](double x) -> double { return x; };
	auto didf = [](double x) -> double { return 1.0; };

    Neuralnet net;
	vector<shared_ptr<Layer>> layers;
	layers.emplace_back(new FullyConnected(1, 1, 1, 100, ReLU, dReLU));
	layers.emplace_back(new FullyConnected(1, 100, 1, 1, idf, didf));

	net.add_layer(layers[0]);
	net.add_layer(layers[1]);
	
	vector<vector<vector<double>>> x(100), y(100);
	for( int i = 0; i < 100; ++i ){
	    x[i] = {{-1.0 + 2.0/99.0*i}};
		y[i] = {{cos(x[i][0][0])}};
	}
	
    net.learning(x, y, 100*10);
	
	vector<vector<vector<double>>> input = {{{0.8}}};
	auto output = net.apply(input);
	printf("Approximation : %.6E\n", output[0][0][0]);
	printf("Correct       : %.6E\n", cos(input[0][0][0]));
}
