cmake_minimum_required(VERSION 3.10) # 建议使用 3.10+ 以更好地支持 CUDA
project(trt_engine LANGUAGES CXX CUDA)

# 1. 设置标准和编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89) 

# 2. 寻找依赖包
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED) 

# 设置 TensorRT 路径
set(TENSORRT_DIR "/home/ubuntu24/下载/TensorRT-8.6.1.6")

# 3. 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/../include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 4. 库目录目录
link_directories(
    ${TENSORRT_DIR}/lib
)

set(SOURCES
    cpp/main.cpp
    cpp/trt_classifier.cpp
    cpp/trt_detector.cpp
    cpp/trt_logger.cpp
    cpp/trt_model.cpp
    cpp/trt_preprocess.cpp
    cpp/trt_preprocess.cu
    cpp/trt_timer.cpp
    cpp/trt_worker.cpp
    cpp/utils.cpp
)

# 6. 生成可执行文件
add_executable(trt_engine ${SOURCES})

# 7. 链接库
target_link_libraries(trt_engine PRIVATE
    CUDA::cudart
    nvinfer
    nvonnxparser
    ${OpenCV_LIBS}
    pthread
    stdc++fs
)
