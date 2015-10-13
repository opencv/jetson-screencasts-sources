cmake_minimum_required(VERSION 2.8)

project(cv_threads)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_threads threads.cpp)

target_link_libraries(cv_threads ${OpenCV_LIBS})
