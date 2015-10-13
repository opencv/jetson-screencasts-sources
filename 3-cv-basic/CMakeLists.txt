cmake_minimum_required(VERSION 2.8)

project(cv_basic)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_basic basic.cpp)

target_link_libraries(cv_basic ${OpenCV_LIBS})
