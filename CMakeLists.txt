cmake_minimum_required(VERSION 3.16)
project(FlashAttn2CPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_library(flash_attn_compile_options INTERFACE)
target_compile_definitions(flash_attn_compile_options INTERFACE EIGEN_NO_DEBUG)
target_compile_options(flash_attn_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -ffast-math -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast /arch:AVX2>
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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

foreach(bench bench_naive bench_fa2 bench_fa2_profile)
    add_executable(${bench} benchmarks/${bench}.cpp)
    target_link_libraries(${bench} PRIVATE flash_attn_st)
endforeach()

add_executable(bench_fa2_mt benchmarks/bench_fa2_mt.cpp)
target_link_libraries(bench_fa2_mt PRIVATE flash_attn_mt)
