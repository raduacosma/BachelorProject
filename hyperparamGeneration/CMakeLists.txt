cmake_minimum_required(VERSION 3.19)
project(hyperparamGeneration)

set(CMAKE_CXX_STANDARD 20)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${hyperparamGeneration_SOURCE_DIR})
add_executable(hyperparamGeneration main.cpp hyperparamSpec.h generateHyperparams.cpp generateHyperparams.h opTypes.h hyperparamSpec.cpp)
