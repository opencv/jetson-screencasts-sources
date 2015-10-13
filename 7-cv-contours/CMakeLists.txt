cmake_minimum_required(VERSION 2.8)

project(cv_contours)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_contours contours.cpp)

target_link_libraries(cv_contours ${OpenCV_LIBS})
