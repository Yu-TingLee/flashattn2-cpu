cmake_minimum_required(VERSION 3.16)
project(FlashAttn2CPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Eigen 5's AVX-512 kernels use intrinsics that GCC only provides from GCC 10.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(FATAL_ERROR "GCC >= 10 is required; configure with -DCMAKE_CXX_COMPILER=g++-11")
endif()

# Header-only, pinned so every build uses the same Eigen. Its CMake project is
# not added, only its headers.
include(FetchContent)
FetchContent_Declare(eigen
    URL      https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    URL_HASH SHA256=e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
endif()
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_library(flash_attn_compile_options INTERFACE)
target_compile_definitions(flash_attn_compile_options INTERFACE EIGEN_NO_DEBUG)
target_compile_options(flash_attn_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -ffast-math -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast /arch:AVX2>
)
# -ffast-math at link time turns on flush-to-zero. Without it, the masked
# naive kernel's exp produces subnormals and runs up to 35% slower.
target_link_options(flash_attn_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffast-math>
)

add_library(flash_attn_io STATIC src/matrix_io.cpp)
target_include_directories(flash_attn_io PUBLIC include)
target_link_libraries(flash_attn_io PUBLIC Eigen3::Eigen flash_attn_compile_options)

# The single-threaded kernels are built without OpenMP so that Eigen does not
# parallelize their matrix products.
add_library(flash_attn_st STATIC
    src/naive_attention.cpp
    src/flash_attention.cpp
    src/flash_attention_profile.cpp
)
target_link_libraries(flash_attn_st PUBLIC flash_attn_io)

add_library(flash_attn_mt STATIC src/flash_attention_mt.cpp)
target_link_libraries(flash_attn_mt PUBLIC flash_attn_io OpenMP::OpenMP_CXX)

add_library(flash_attn_causal_mt STATIC src/flash_attention_causal_mt.cpp)
target_link_libraries(flash_attn_causal_mt PUBLIC flash_attn_st OpenMP::OpenMP_CXX)
target_link_libraries(flash_attn_causal_mt PRIVATE Threads::Threads)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(bench bench_naive bench_naive_causal bench_fa2 bench_fa2_causal bench_fa2_profile)
    add_executable(${bench} benchmarks/${bench}.cpp)
    target_link_libraries(${bench} PRIVATE flash_attn_st)
endforeach()

add_executable(bench_fa2_mt benchmarks/bench_fa2_mt.cpp)
target_link_libraries(bench_fa2_mt PRIVATE flash_attn_mt)

add_executable(bench_fa2_causal_mt benchmarks/bench_fa2_causal_mt.cpp)
target_link_libraries(bench_fa2_causal_mt PRIVATE flash_attn_causal_mt)
