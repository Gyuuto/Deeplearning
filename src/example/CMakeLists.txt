cmake_minimum_required(VERSION 2.8)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if( NOT DEFINED ${CMAKE_CXX_FLAGS} )
  set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall")

  find_package(OpenMP)
  if( OpenMP_FOUND )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()
message(STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(BLAS)
if( BLAS_FOUND )
  add_definitions(-DUSE_BLAS)
else()
  message(WARNING "No BLAS is found. Use simple implementation instead of it.")
endif()

add_subdirectory(approx_cosine)
add_subdirectory(mnist_sample)
add_subdirectory(mnist_sample_dist)
add_subdirectory(mnist_sample_gpu)
