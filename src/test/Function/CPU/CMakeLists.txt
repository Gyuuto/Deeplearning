cmake_minimum_required(VERSION 2.8)

add_executable(softmaxTest_CPU_float softmaxTest_CPU_float.cpp)
target_link_libraries(softmaxTest_CPU_float ${GTEST_LIBRARIES})
target_link_libraries(softmaxTest_CPU_float ${BLAS_LIBRARIES})

add_executable(crossentropyTest_CPU_float crossentropyTest_CPU_float.cpp)
target_link_libraries(crossentropyTest_CPU_float ${BLAS_LIBRARIES})
target_link_libraries(crossentropyTest_CPU_float ${GTEST_LIBRARIES})

add_executable(reluTest_CPU_float reluTest_CPU_float.cpp)
target_link_libraries(reluTest_CPU_float ${BLAS_LIBRARIES})
target_link_libraries(reluTest_CPU_float ${GTEST_LIBRARIES})
