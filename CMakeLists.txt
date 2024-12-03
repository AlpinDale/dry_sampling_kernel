cmake_minimum_required(VERSION 3.18)
project(dry_sampling_kernel CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_86")  # this is for my GPU

add_executable(dry_sampling_kernel
    dry.cu
    dry_test.cu
)

target_link_libraries(dry_sampling_kernel
    ${CUDA_LIBRARIES}
    ${TORCH_LIBRARIES}
    cuda
)

target_include_directories(dry_sampling_kernel PRIVATE
    ${TORCH_INCLUDE_DIRS}
)
