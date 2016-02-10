# Neural network C++ Library
----
Programs of relation with artificial neural network. There are 4 types of layers, FullyConnected, SparseFullyConnected, Convolutional and Max-Pooling. I introduce these layers in below.


# Layers
----
* FullyConnected
    
  FullyConnected layer is a hidden layer of multi layer paceptron. There are input units and output units and they are connected each other. This connection is expressed by matrix which has output units multiply input units plus 1 elements.
  
* SparseFullyConnected
  
  SparseFullyConnected is similar with FullyConnected but this layer attempt to be sparse the output units by Kullback-Leibler divergence. So this layer is required activate function that always be plus value.
  
* Convolutional

  Convolutional is to do convolution operation to input image or something(I assume that input is image). This layer has zero padding only and a stride isn't working now.

* Max-Pooling

  Max-Pooling is to do pooling by maximum value in some window.
  
# Usage
----
### Approximation cosine function

	// define some function and differential.
	auto ReLU = [](double x) -> double { return max(0.0, x); };
	auto dReLU = [](double x) -> double { return (x <= 0.0 ? 0.0 : 1.0); };
	auto idf = [](double x) -> double { return x; };
	auto didf = [](double x) -> double { return 1.0; };

    Neuralnet net;
	vector<shared_ptr<Layer>> layers;
	// define layers.
	layers.emplace_back(new FullyConnected(1, 1, 1, 100, ReLU, dReLU));
	layers.emplace_back(new FullyConnected(1, 100, 1, 1, idf, didf));

	// add layers to the neuralnet
	net.add_layer(layers[0]);
	net.add_layer(layers[1]);
	
	// set input values and supervised values
	vector<vector<vector<double>>> x(100), y(100);
	for( int i = 0; i < 100; ++i ){
	    x[i] = {{-1.0 + 2.0/99.0*i}};
		y[i] = {{cos(x[i][0][0])}};
	}
	
	// learning the neuralnet with 10 EPOCH.
    net.learning(x, y, 100*10);
	
	// check a approximate function by neuralnet.
	vector<vector<vector<double>>> input = {{{0.8}}};
	auto output = net.apply(input);
	printf("Approximation : %.6E\n", output[0][0][0]);
	printf("Correct       : %.6E\n", cos(input[0][0][0]));

