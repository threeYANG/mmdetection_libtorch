cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DetectorAPI LANGUAGES CXX C)

SET(CMAKE_BUILD_TYPE "debug")


# set cuda path
set(LIB_PATH "/home/dl/3rdparty/")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(Torch_DIR "${LIB_PATH}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install")
message("install prefix: ${CMAKE_INSTALL_PREFIX}")


add_subdirectory(src)
add_subdirectory(example)

