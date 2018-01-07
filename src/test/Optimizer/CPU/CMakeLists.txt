cmake_minimum_required(VERSION 2.8)

add_executable(adamTest_CPU_float adamTest_CPU_float.cpp)
target_link_libraries(adamTest_CPU_float ${GTEST_LIBRARIES})
target_link_libraries(adamTest_CPU_float ${BLAS_LIBRARIES})
