# Neural network C++ Library
----
Programs of relation with artificial neural network. There are 4 types of layers, FullyConnected, SparseFullyConnected, Convolutional and Max-Pooling. I introduce these layers in below.
This library is released under the MIT License, for more details see a file LICENSE.

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

Check example folder. There are 2 files, approximation cosine function and learning hand written data MNIST(http://yann.lecun.com/exdb/mnist/) with Convolutional Neural Network.
