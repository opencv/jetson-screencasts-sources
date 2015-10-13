cmake_minimum_required(VERSION 2.8)

project(cv_mono_calibrate)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_mono_calibrate mono_calibrate.cpp)

target_link_libraries(cv_mono_calibrate ${OpenCV_LIBS})
