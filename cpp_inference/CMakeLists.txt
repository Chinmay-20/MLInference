cmake_minimum_required(VERSION 3.20)
project(ONNXInference CXX)

# ------------------ Standard C++ ------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ ONNX Runtime ------------------
# Path priority:
# 1) -DONNXRUNTIME_DIR passed manually or from CI
# 2) macOS Homebrew default

if(DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_ROOT "${ONNXRUNTIME_DIR}")
    message(STATUS "Using ONNX Runtime from ONNXRUNTIME_DIR=${ONNXRUNTIME_ROOT}")
else()
    set(ONNXRUNTIME_ROOT "/opt/homebrew/opt/onnxruntime")
    message(STATUS "Using default ONNX Runtime path: ${ONNXRUNTIME_ROOT}")
endif()

# Find headers
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime
)

# Find library
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found.\n"
        "Set -DONNXRUNTIME_DIR=/path/to/onnxruntime"
    )
endif()

# ------------------ YAML-CPP ------------------
find_package(yaml-cpp REQUIRED)

if(TARGET yaml-cpp::yaml-cpp)
    set(YAMLCPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
    set(YAMLCPP_TARGET yaml-cpp)
else()
    message(FATAL_ERROR "yaml-cpp target not found")
endif()


# ------------------ Executable ------------------
add_executable(onnx_inference
    src/main.cpp
)

# Include directories
target_include_directories(onnx_inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

# Link libraries
target_link_libraries(onnx_inference
    ${ONNXRUNTIME_LIB}
    ${YAMLCPP_TARGET}
)
