cmake_minimum_required(VERSION 2.8)

project(cv_stereo_calibrate)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cv_stereo_calibrate stereo_calibrate.cpp)

target_link_libraries(cv_stereo_calibrate ${OpenCV_LIBS})
