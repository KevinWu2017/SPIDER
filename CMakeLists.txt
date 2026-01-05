cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(SPTCConv LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)

# Define the include DIRs
include_directories(/usr/local/cuda/include)
include_directories(third_party/argparse/include)

# Define the link libraries
link_directories(/usr/local/cuda/lib64)

# 设置可执行文件的输出目录为 build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src/1d_half_sparse)
add_subdirectory(src/2d_half_dense)
add_subdirectory(src/2d_half_sparse)
add_subdirectory(src/2d_half_sparse_for_ablation)
