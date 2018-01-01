cmake_minimum_required(VERSION 2.8)

add_executable(approx_cosine approx_cosine.cpp)
target_link_libraries(approx_cosine ${BLAS_LIBRARIES})

