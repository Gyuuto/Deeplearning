# Neural network C++ Library
----
Library of relation with artificial neural network written by C++. There are 6 types of layers, FullyConnected, SparseFullyConnected, DropoutFullyConnected, KDropoutFullyConnected, Convolutional and Max-Pooling. I introduce these layers in below.
This library is released under the MIT License, for more details see a file LICENSE.

# Layers
----
* FullyConnected
    
  FullyConnected layer is a hidden layer of multi layer paceptron. There are input units and output units and they are connected each other. This connection is expressed by matrix which has output units multiply input units plus 1 elements.
  
* SparseFullyConnected
  
  SparseFullyConnected is similar with FullyConnected but this layer attempt to be sparse the output units by Kullback-Leibler divergence. So this layer is required activate function that always be plus value.
  
* DropoutFullyConnected
  
  DropoutFullyConnected is FullyConnected layer with dropout. Dropout regards connection between output unit and intput unit is lost. Dropout improves generalization ability but slow donw to convergence. This library determine how to loose a connection by uniform random number in given probability.

* KDropoutFullyConnected
  
  KDropoutFullyConnected is alse FullyConnected layer with dropout but how to loose connection is difference above one. This layer determine it by choosing some maximum activation values. K maximum activation values are going through next layer but other activation set to zero.
  
* Convolutional

  Convolutional is to do convolution operation to input image or something(I assume that input is image). This layer has zero padding only and a stride isn't working now.

* Max-Pooling

  Max-Pooling is to do pooling by maximum value in some window.
  
# Usage
----

Check example folder. There are 3 files, approximation cosine function, learning hand written data MNIST(http://yann.lecun.com/exdb/mnist/) with Convolutional Neural Network and distributed version of it.

