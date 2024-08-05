cmake_minimum_required(VERSION 3.17)
project(cudnn_frontend_example LANGUAGES CXX CUDA)

include(${PROJECT_SOURCE_DIR}/cuDNN.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-error=attributes -Wno-attributes -Wno-error=unused-function -Wno-unused-function)

add_library(cudnn_frontend INTERFACE)

target_include_directories(cudnn_frontend INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(
    cudnn_frontend INTERFACE
    CUDA::cudart
    CUDA::nvrtc
)

target_compile_features(cudnn_frontend INTERFACE cxx_std_17)

add_library(_cudnn_frontend_pch INTERFACE)
target_precompile_headers(_cudnn_frontend_pch INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include/cudnn_frontend.h
)

add_executable(cudnn_frontend_example src/main.cpp)

target_link_libraries(cudnn_frontend_example PRIVATE cudnn_frontend _cudnn_frontend_pch CUDNN::cudnn_all)
target_include_directories(cudnn_frontend_example PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)
