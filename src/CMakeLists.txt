cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DetectorModule  LANGUAGES CXX C CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3  -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D__CUDA_NO_HALF_OPERATORS__")

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_INCLUDE_DIRS" ${CUDA_INCLUDE_DIRS})


#opencv
set(OpenCV_DIR "/usr/share/OpenCV")
find_package(OpenCV  REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 项目头文件目录
set(PROJECT_INCLUDES
    ./
    ../include/
    anchor_point/
    bbox/
    nms/
    detector/
    transform/
    utils/
    single_roi_extractor/
    roi_align/
    cuda_helper/
)

foreach(PROJECT_INCLUDE ${PROJECT_INCLUDES})
    include_directories(${PROJECT_SOURCE_DIR}/${PROJECT_INCLUDE})
endforeach(PROJECT_INCLUDE)

set(SRC
    ./Detector.cpp
    ./DetectorImpl.cpp
    ./types.cpp
    ./DetectorCreator.cpp
    ./anchor_point/AnchorPointGenerator.cpp
    ./bbox/bbox.cpp
    ./nms/nms_cuda.cu
    ./nms/nms.cpp
    ./single_roi_extractor/SingleRoIExtractor.cpp
    ./roi_align/RoIAlign.cpp
    ./roi_align/roi_align_cuda.cu

    #transforms
    ./transform/transforms.cpp

    #detector
    ./detector/DetectorSSD.cpp
    ./detector/DetectorFCOS.cpp
    ./detector/DetectorRetinanet.cpp
    ./detector/DetectorFasterMaskRcnn.cpp
    ./detector/DetectorCommon.cpp

    #utils
    ./utils/logging.hpp
    ./utils/timer.hpp
    ./utils/jsoncpp.cpp
    )

set(DEP_LIBS  ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# 编译库, 输出到 ${CMAKE_BINARY_DIR}/lib
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(DetectorModule SHARED ${SRC})
target_link_libraries(DetectorModule ${DEP_LIBS})
set_target_properties(DetectorModule PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

################################### 安装配置 ########################################
install (TARGETS DetectorModule
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

