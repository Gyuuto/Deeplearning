cmake_minimum_required(VERSION 2.8)

include_directories(src/include)
add_subdirectory(src/example)
if( NOT without-test )
  enable_testing()
  find_package(GTest)

  add_subdirectory(src/test)
endif()
