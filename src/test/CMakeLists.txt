cmake_minimum_required(VERSION 2.8)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
if( NOT DEFINED ${CMAKE_CXX_FLAGS} )
  set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -Wall")

  find_package(OpenMP)
  if( OpenMP_FOUND )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()
message(STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")

include_directories(${GTEST_INCLUDE_DIRS})
add_subdirectory(Layer/CPU)
add_subdirectory(Layer/OpenCL)
add_subdirectory(Layer/CUDA)

add_subdirectory(Function/CPU)
add_subdirectory(Function/OpenCL)
add_subdirectory(Function/CUDA)

add_subdirectory(Optimizer/CPU)
add_subdirectory(Optimizer/OpenCL)
add_subdirectory(Optimizer/CUDA)

add_subdirectory(Matrix)
