cmake_minimum_required(VERSION 3.5)

project(MCSPH LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}") 

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mtune=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
cmake_policy(SET CMP0104 NEW) # for CUDA_ARCHITECTURES
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG --extended-lambda --expt-relaxed-constexpr -std=c++17")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)
# Add the path to the OptiX headers to our include paths.
include_directories(${OptiX_INCLUDE})

add_subdirectory(3rd)
add_subdirectory(Utils)
add_subdirectory(MCISPH)