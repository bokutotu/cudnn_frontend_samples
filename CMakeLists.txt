cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 20)
project(cudnn_frontend_example LANGUAGES CXX CUDA)

include(${PROJECT_SOURCE_DIR}/cuDNN.cmake)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-error=attributes -Wno-attributes -Wno-error=unused-function -Wno-unused-function)

add_library(cudnn_frontend INTERFACE)

target_include_directories(cudnn_frontend INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)

find_package(CUDAToolkit REQUIRED)

find_package(Threads)

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)

    # Fetch and build catch2
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.3.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

target_link_libraries(
    cudnn_frontend INTERFACE
    CUDA::cudart
    CUDA::nvrtc
    Threads::Threads
    Catch2::Catch2WithMain
)

target_compile_features(cudnn_frontend INTERFACE cxx_std_17)

add_library(_cudnn_frontend_pch INTERFACE)
target_precompile_headers(_cudnn_frontend_pch INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include/cudnn_frontend.h
)

# add_executable(
  # cudnn_frontend_example 
  # src/main.cpp 
  # src/conv.cpp 
  # src/batch_norm.cpp
# )

add_executable(
  cudnn_batch_norm_bench 
  src/bench_batch_norm.cpp
)

add_executable(
  cudnn_batch_norm_2d_bench
  src/bench_batch_norm_2d.cpp
)
# target_link_libraries(cudnn_frontend_example PRIVATE cudnn_frontend _cudnn_frontend_pch CUDNN::cudnn_all)
# target_include_directories(cudnn_frontend_example PRIVATE 
#     ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
# )

target_link_libraries(cudnn_batch_norm_bench PRIVATE cudnn_frontend _cudnn_frontend_pch CUDNN::cudnn_all)
target_include_directories(cudnn_batch_norm_bench PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)

target_link_libraries(cudnn_batch_norm_2d_bench PRIVATE cudnn_frontend _cudnn_frontend_pch CUDNN::cudnn_all)
target_include_directories(cudnn_batch_norm_2d_bench PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)
