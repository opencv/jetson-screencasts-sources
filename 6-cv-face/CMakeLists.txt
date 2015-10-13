cmake_minimum_required(VERSION 2.8)

project(cv_face)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_face face.cpp)

target_link_libraries(cv_face ${OpenCV_LIBS})
