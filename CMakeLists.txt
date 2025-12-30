cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(unidepth_tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(BUILD_EXAMPLES "Build example applications" ON)
option(ENABLE_CUDA_GRAPH "Enable CUDA Graph optimization" ON)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT
if(DEFINED ENV{TensorRT_ROOT})
    set(TensorRT_ROOT $ENV{TensorRT_ROOT})
elseif(DEFINED TensorRT_ROOT)
    # Use the provided TensorRT_ROOT
else()
    set(TensorRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation directory")
endif()

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES include)

find_library(TensorRT_LIBRARY nvinfer
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TensorRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

find_library(TensorRT_ONNX_PARSER_LIBRARY nvonnxparser
    HINTS ${TensorRT_ROOT}
    PATH_SUFFIXES lib lib64)

if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    set(TensorRT_FOUND TRUE)
    message(STATUS "Found TensorRT: ${TensorRT_LIBRARY}")
    message(STATUS "  Include: ${TensorRT_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "TensorRT not found. Please set TensorRT_ROOT or install TensorRT")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIR}
)

# CUDA architectures (RTX 20xx-50xx)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# CUDA kernels library
add_library(unidepth_kernels STATIC
    kernels/preprocessing.cu
    kernels/postprocessing.cu
)

target_link_libraries(unidepth_kernels
    CUDA::cudart
    ${OpenCV_LIBS}
)

# Core library
add_library(unidepth_core STATIC
    src/unidepth_v2.cpp
)

target_link_libraries(unidepth_core
    unidepth_kernels
    ${TensorRT_LIBRARY}
    ${TensorRT_PLUGIN_LIBRARY}
    ${TensorRT_ONNX_PARSER_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)

# Example: Image inference
if(BUILD_EXAMPLES)
    add_executable(image_inference
        examples/image_inference.cpp
    )

    target_link_libraries(image_inference
        unidepth_core
        unidepth_kernels
        ${TensorRT_LIBRARY}
        CUDA::cudart
        ${OpenCV_LIBS}
    )

    # Install
    install(TARGETS image_inference
        RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build/bin
    )
endif()

# Print configuration
message(STATUS "")
message(STATUS "==============================================")
message(STATUS "UniDepth TensorRT Configuration")
message(STATUS "==============================================")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Compiler:       ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA Compiler:      ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT:           ${TensorRT_LIBRARY}")
message(STATUS "OpenCV:             ${OpenCV_VERSION}")
message(STATUS "CUDA Graph:         ${ENABLE_CUDA_GRAPH}")
message(STATUS "Build Examples:     ${BUILD_EXAMPLES}")
message(STATUS "==============================================")
message(STATUS "")
