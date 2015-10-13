cmake_minimum_required(VERSION 2.8)

project(cv_mat)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_mat mat.cpp)

target_link_libraries(cv_mat ${OpenCV_LIBS})
