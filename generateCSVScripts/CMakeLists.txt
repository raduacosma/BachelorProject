cmake_minimum_required(VERSION 3.19)
project(generateProperCSV)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${generateProperCSV_SOURCE_DIR})
add_executable(generateProperCSV main.cpp)