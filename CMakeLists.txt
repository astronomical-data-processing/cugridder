cmake_minimum_required(VERSION 3.10.2)

#! Base project
project(curig C CXX CUDA)

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Include source code build process
include_directories(include include/cuda_sample)
add_subdirectory("test")

# Define optimization flag based on build type (DEBUG/RELEASE)
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-fPIC -Wall -Wextra -funroll-loops -march=native")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra -std=c++14")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")    
endif()
