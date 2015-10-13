cmake_minimum_required(VERSION 2.8)

project(cv_stereo)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_stereo stereo.cpp)

target_link_libraries(cv_stereo ${OpenCV_LIBS})
