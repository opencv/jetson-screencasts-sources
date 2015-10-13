cmake_minimum_required(VERSION 2.8)

project(cv_flow)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_flow flow.cpp)

target_link_libraries(cv_flow ${OpenCV_LIBS})
