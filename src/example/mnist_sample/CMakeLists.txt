cmake_minimum_required(VERSION 2.8)

add_executable(mnist_sample mnist_sample.cpp)
target_link_libraries(mnist_sample ${BLAS_LIBRARIES})

