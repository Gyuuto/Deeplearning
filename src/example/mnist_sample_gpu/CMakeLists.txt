cmake_minimum_required(VERSION 2.8)

find_package(OpenCL)
include(/usr/local/lib64/cmake/clBLAS/clBLASConfig.cmake)

if( OpenCL_FOUND AND CLBLAS_LIBRARIES )
  # set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall -fopenmp")
  # set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  add_executable(mnist_sample_gpu mnist_sample_gpu.cpp)
  add_definitions(-DUSE_GPU)
  include_directories(${OpenCL_INCLUDE_DIRS})
  target_link_libraries(mnist_sample_gpu ${BLAS_LIBRARIES})
  target_link_libraries(mnist_sample_gpu ${OpenCL_LIBRARIES})
  target_link_libraries(mnist_sample_gpu ${CLBLAS_LIBRARIES})
else()
  message(STATUS "No OpenCL is found. Can't build mnist_sample_gpu.")
endif()

