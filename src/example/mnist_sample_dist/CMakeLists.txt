cmake_minimum_required(VERSION 2.8)

find_package(MPI)

if( MPI_FOUND )
  set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")

  add_executable(mnist_sample_dist mnist_sample_dist.cpp)
  include_directories(${MPI_INCLUDE_PATH})
  add_definitions(-DUSE_MPI)
  target_link_libraries(mnist_sample_dist ${BLAS_LIBRARIES})
  target_link_libraries(mnist_sample_dist ${MPI_LIBRARIES})
else()
  message(STATUS "No installed MPI is found. Can't build mnist_sample_dist.")
endif()
