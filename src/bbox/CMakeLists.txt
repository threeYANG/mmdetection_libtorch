cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nms_cuda)

find_package(CUDA)
include_directories(/usr/local/cuda/include/)
link_directories(/usr/local/cuda/lib64)

#libtorch
set(LIB_PATH "/home/dl/3rdparty/")
set (Torch_DIR "${LIB_PATH}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message("TORCH_INCLUDE_DIRS " ${TORCH_INCLUDE_DIRS})

add_definitions(-DGPU_CUDA)
SET(CUDA_NVCC_FLAGS -Xcompiler -std=c++11 -Xcompiler -fPIC)


#cuda_add_library(nms STATIC
cuda_add_library(nms SHARED
  nms_kernel.cu
  nms_kernel.h
)


