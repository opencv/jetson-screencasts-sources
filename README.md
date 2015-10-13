# About

This repository contains the sources of the examples from screencasts about
OpenCV programming on [NVidia Jetson platform](https://developer.nvidia.com/jetson-tk1).

# Prerequisites

- All the examples are built with the help of [CMake](https://cmake.org/) build
  system. To install it, run the following command in the terminal:
```
  $ sudo apt-get install cmake
```
- The only dependency of the examples is the OpenCV4Tegra library. It should be
  installed on the platform as part of the
  [JetPack](https://developer.nvidia.com/jetson-tk1-development-pack)
  package. If, for some reason, you don't have it installed on the system, you
  can try the following command:
```
  $ sudo apt-get install libopencv4tegra-dev
```

# Building

Each example is self-contained and is located in a separate folder. All of them
can be built using the same procedure:

- Go to the directory of the sample
```
$ cd ~/0-cv-hello
```
- Create build directory and go to it
```
$ mkdir build
$ cd build
```
- Run cmake to generate a Makefile
```
$ cmake ..
```
- Run make to build the executable
```
$ make
```
- Go one level up (so that the application can locate the media files) and run
  the executable
```
$ cd ..
$ ./build/cv_hello
```

# Notes

- Examples `11-cv-perf`,  `12-cv-threads`, `13-cv-neon`, `14-cv-gpu` use C++11
  features
- Examples `12-cv-threads` and `13-cv-neon` use OpenMP.
- Example `13-cv-neon` uses NEON compiler intrinsics.
