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

# Shared compile options
add_library(flash_compile_opts INTERFACE)
target_compile_definitions(flash_compile_opts INTERFACE EIGEN_NO_DEBUG)
target_compile_options(flash_compile_opts INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -ffast-math -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast /arch:AVX2>
)

# I/O utility library
# Provides: load_npy, load_matrix
# PUBLIC deps are forwarded to every library/executable that links this target.
add_library(flash_cpu_io STATIC src/io.cpp)
target_include_directories(flash_cpu_io PUBLIC include)
target_link_libraries(flash_cpu_io PUBLIC Eigen3::Eigen flash_compile_opts)

# Single-threaded algorithm library
# Provides: naive_attention, fa2_forward (ST), fa2_forward_profile
add_library(flash_cpu_st STATIC
    src/attn.cpp
    src/attn_profile.cpp
)
target_link_libraries(flash_cpu_st PUBLIC flash_cpu_io)

# Multi-threaded algorithm library
# Provides: fa2_forward (MT, OpenMP)
# with the ST version. bench_fa2_mt links this; all other tools link flash_cpu_st.
add_library(flash_cpu_mt STATIC src/attn_mt.cpp)
target_link_libraries(flash_cpu_mt PUBLIC flash_cpu_io OpenMP::OpenMP_CXX)

# Multi-threaded causal FA-2 library
# Provides: fa2_forward_causal_mt (static/dynamic/work-stealing)
# Also provides: naive_attention_causal, fa2_forward_causal (from flash_cpu_st)
add_library(flash_cpu_mt_causal STATIC src/attn_mt_causal.cpp)
target_link_libraries(flash_cpu_mt_causal PUBLIC flash_cpu_st OpenMP::OpenMP_CXX)
target_link_libraries(flash_cpu_mt_causal PRIVATE pthread)

# Benchmark executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(bench_naive_O3            tools/bench_naive.cpp)
add_executable(bench_naive_causal_O3     tools/bench_naive_causal.cpp)
add_executable(bench_fa2_O3             tools/bench_fa2.cpp)
add_executable(bench_fa2_profile_O3     tools/bench_fa2_profile.cpp)
add_executable(bench_fa2_O3_mt          tools/bench_fa2_mt.cpp)
add_executable(bench_fa2_causal_mt_O3   tools/bench_fa2_causal_mt.cpp)

target_link_libraries(bench_naive_O3          PRIVATE flash_cpu_st)
target_link_libraries(bench_naive_causal_O3   PRIVATE flash_cpu_st)
target_link_libraries(bench_fa2_O3           PRIVATE flash_cpu_st)
target_link_libraries(bench_fa2_profile_O3   PRIVATE flash_cpu_st)
target_link_libraries(bench_fa2_O3_mt        PRIVATE flash_cpu_mt)
target_link_libraries(bench_fa2_causal_mt_O3 PRIVATE flash_cpu_mt_causal)
