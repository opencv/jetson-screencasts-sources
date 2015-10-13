cmake_minimum_required(VERSION 2.8)

project(cv_io)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_io io.cpp)

target_link_libraries(cv_io ${OpenCV_LIBS})
