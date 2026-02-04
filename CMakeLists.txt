cmake_minimum_required(VERSION 3.16)
project(FlashAttn2CPU LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

function(add_fa_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
    target_include_directories(${target_name} PRIVATE ${EIGEN3_INCLUDE_DIR})
    target_compile_definitions(${target_name} PRIVATE EIGEN_NO_DEBUG)
    target_compile_options(${target_name}
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -ffast-math -march=native>
            $<$<CXX_COMPILER_ID:MSVC>:/O2 /fp:fast /arch:AVX2>
    )
endfunction()

add_fa_executable(naive_attn_O3 naive_attn.cpp)
add_fa_executable(flash_attn2_O3 flash_attn2.cpp)
add_fa_executable(flash_attn2_profile_O3 profile/flash_attn2_profile.cpp)
