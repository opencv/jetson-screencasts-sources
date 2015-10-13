cmake_minimum_required(VERSION 2.8)

project(cv_gpu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_gpu gpu.cpp)

target_link_libraries(cv_gpu ${OpenCV_LIBS})
