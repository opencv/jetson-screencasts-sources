cmake_minimum_required(VERSION 2.8)

project(cv_perf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_perf perf.cpp)

target_link_libraries(cv_perf ${OpenCV_LIBS})
