cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

if( CUDA_FOUND )
  set(CUDA_NVCC_FLAGS "-O3 -Xcompiler -Wall")
  include_directories(${CUDA_INCLUDE_DIRS})
  
  add_executable(cudaMatrixTest cudaMatrixTest.cpp)

  file(GLOB_RECURSE CUDA_SRC ${CMAKE_SOURCE_DIR}/src/include/CUDA/*.cu)
  
  cuda_add_library(cuda_kernels ${CUDA_SRC})
  cuda_add_cublas_to_target(cudaMatrixTest)
  target_link_libraries(cudaMatrixTest ${GTEST_LIBRARIES})
  target_link_libraries(cudaMatrixTest cuda_kernels)
else()
  message(STATUS "No CUDA is found. Can't build GPU tests.")
endif()
