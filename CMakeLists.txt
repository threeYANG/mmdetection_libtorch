cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DetectorAPI)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#opencv
set(OpenCV_DIR "/usr/share/OpenCV")
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV_DIR " ${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

#libtorch
set(LIB_PATH "/home/dl/3rdparty/")
set (Torch_DIR "${LIB_PATH}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
message("TORCH_INCLUDE_DIRS " ${TORCH_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include/)
link_directories(/usr/local/cuda/lib64)

#nms_cuda
link_directories(/home/dl/Project/mmdetection_libtorch/src/nms_cuda_lib/)

set(DetectorSrc
        #External Interface
        src/Detector.cpp
        src/DetectorImpl.cpp
        src/types.cpp
        src/DetectorCreator.cpp
        src/anchor/AnchorGenerator.cpp
        src/bbox/bbox.cpp
        #nms
        src/bbox/nms_kernel.h

        #transforms
        src/transform/transforms.cpp

        #detector
        src/detector/DetectorSSD.cpp
        src/detector/DetectorRetinanet.cpp
        src/detector/SingleStage.cpp


        #utils
        src/utils/logging.hpp
        src/utils/timer.hpp
        src/utils/jsoncpp.cpp

        test_detector)

set(DetectorInclude
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/utils
        ${PROJECT_SOURCE_DIR}/src/detector
        ${PROJECT_SOURCE_DIR}/src/anchor
        ${PROJECT_SOURCE_DIR}/src/bbox
        ${PROJECT_SOURCE_DIR}/src/transform
        )

include_directories(${DetectorInclude})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
MESSAGE("TORCH_LIBRARIES " ${TORCH_LIBRARIES})
MESSAGE("OpenCV_LIBRARIES " ${OpenCV_LIBRARIES})
MESSAGE("CUDA_LIBRARIES " ${CUDA_LIBRARIES})


add_executable(DetectorAPI ${DetectorSrc})
set(DEP_LIBS ${OpenCV_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(DetectorAPI ${DEP_LIBS} cuda cudart cublas curand cudnn nms)
set_target_properties(DetectorAPI PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

