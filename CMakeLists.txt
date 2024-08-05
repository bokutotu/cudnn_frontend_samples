cmake_minimum_required(VERSION 3.17)
project(cudnn_frontend_example LANGUAGES CXX CUDA)

# コンパイルオプションの設定
add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-error=attributes -Wno-attributes -Wno-error=unused-function -Wno-unused-function)

# cudnn_frontendライブラリの定義
add_library(cudnn_frontend INTERFACE)

# インクルードディレクトリの設定
target_include_directories(cudnn_frontend INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cudnn_frontend/include)

# CUDAToolkitの検索
find_package(CUDAToolkit REQUIRED)

# ライブラリのリンク
target_link_libraries(
    cudnn_frontend INTERFACE
    CUDA::cudart
    CUDA::nvrtc
)

# C++17の使用を指定
target_compile_features(cudnn_frontend INTERFACE cxx_std_17)

# プリコンパイル済みヘッダーの設定
add_library(_cudnn_frontend_pch INTERFACE)
target_precompile_headers(_cudnn_frontend_pch INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cudnn_frontend/include/cudnn_frontend.h)

# 実行ファイルの追加
add_executable(cudnn_frontend_example src/main.cpp)

# 実行ファイルにcudnn_frontendをリンク
target_link_libraries(cudnn_frontend_example PRIVATE cudnn_frontend)
