set(LIB_PATH "/home/dl/3rdparty/")

#cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(${CUDA_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
set(EXAMPLE_BIN_PATH ${CMAKE_BINARY_DIR}/)

set(DEP_LIBS DetectorModule)

add_executable(test_detect test_detector.cpp)
set_target_properties(test_detect PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BIN_PATH})
target_link_libraries(test_detect ${DEP_LIBS})

