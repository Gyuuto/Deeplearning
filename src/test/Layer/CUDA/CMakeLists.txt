cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

if( CUDA_FOUND )
  include_directories(${CUDA_INCLUDE_DIRS})

  add_executable(FullyConnectedTest_CUDA_float FullyConnectedTest_CUDA_float.cpp)
  add_definitions(-DUSE_CUDA)
  cuda_add_cublas_to_target(FullyConnectedTest_CUDA_float)
  target_link_libraries(FullyConnectedTest_CUDA_float ${BLAS_LIBRARIES})
  target_link_libraries(FullyConnectedTest_CUDA_float ${GTEST_LIBRARIES})
  target_link_libraries(FullyConnectedTest_CUDA_float cuda_kernels)

  # add_executable(BatchNormalize_OpenCL_float BatchNormalize_OpenCL_float.cpp)
  # add_definitions(-DUSE_OPENCL)
  # include_directories(${OpenCL_INCLUDE_DIRS})
  # target_link_libraries(BatchNormalize_OpenCL_float ${BLAS_LIBRARIES})
  # target_link_libraries(BatchNormalize_OpenCL_float ${GTEST_LIBRARIES})
  # target_link_libraries(BatchNormalize_OpenCL_float ${OpenCL_LIBRARIES})
  # target_link_libraries(BatchNormalize_OpenCL_float ${CLBLAS_LIBRARIES})
else()
  message(STATUS "No CUDA is found. Can't build CUDA tests.")
endif()
