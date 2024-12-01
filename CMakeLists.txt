cmake_minimum_required(VERSION 3.18)

project(cudnn_frontend_sample VERSION 1.8.0)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
    include(FetchContent)

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.3.2
    )
    FetchContent_MakeAvailable(Catch2)
endif()

include(${PROJECT_SOURCE_DIR}/cudnn-frontend/cmake/cuDNN.cmake)

if(DEFINED ENV{NO_DEFAULT_IN_SWITCH})
    message("Default case in the switch is disabled")
    add_compile_definitions(NO_DEFAULT_IN_SWITCH)
endif()

add_executable(samples
  src/bench_batch_norm_2d.cpp
)

# target_include_directories(samples PRIVATE ${CMAKE_SOURCE_DIR}/cudnn-frontend/include)
add_library(cudnn_frontend INTERFACE)
target_include_directories(cudnn_frontend INTERFACE ${CMAKE_SOURCE_DIR}/cudnn-frontend/include)
target_compile_definitions(
    cudnn_frontend INTERFACE
    $<$<BOOL:${CUDNN_FRONTEND_SKIP_JSON_LIB}>:CUDNN_FRONTEND_SKIP_JSON_LIB>
)
target_include_directories(
    cudnn_frontend INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(cudnn_frontend INTERFACE cxx_std_17)
add_library(_cudnn_frontend_pch INTERFACE)
target_precompile_headers(_cudnn_frontend_pch INTERFACE ${PROJECT_SOURCE_DIR}/cudnn-frontend/include/cudnn_frontend.h)

target_compile_options(
  samples PRIVATE
  -Wall
  -Wextra
  -Werror
  -Wno-unused-function
)

target_link_libraries(
  samples PRIVATE
  Threads::Threads
  Catch2::Catch2WithMain
  cudnn_frontend
  _cudnn_frontend_pch
  CUDNN::cudnn
  CUDA::cudart
  CUDA::cuda_driver
)
