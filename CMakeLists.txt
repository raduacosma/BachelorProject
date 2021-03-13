cmake_minimum_required(VERSION 3.15)
project(BachelorProject)
INCLUDE(FindPkgConfig)
pkg_search_module(GLFW REQUIRED glfw3)
include_directories(${GLFW_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

include_directories(imgui-1.81)
include_directories(imgui-1.81/backends)
include_directories(libs)
include_directories(libs/gl3w)
include_directories(libs/gl3w/GL)
include_directories(libs/glfw)
include_directories(libs/glfw/include)
include_directories(libs/glfw/include/GLFW)

add_executable(BachelorProject
        imgui-1.81/backends/imgui_impl_glfw.cpp
        imgui-1.81/backends/imgui_impl_glfw.h
        imgui-1.81/backends/imgui_impl_opengl3.cpp
        imgui-1.81/backends/imgui_impl_opengl3.h
        imgui-1.81/imconfig.h
        imgui-1.81/imgui.cpp
        imgui-1.81/imgui.h
        imgui-1.81/imgui_demo.cpp
        imgui-1.81/imgui_draw.cpp
        imgui-1.81/imgui_internal.h
        imgui-1.81/imgui_tables.cpp
        imgui-1.81/imgui_widgets.cpp

        libs/gl3w/GL/gl3w.c
        libs/gl3w/GL/gl3w.h
        libs/gl3w/GL/glcorearb.h
        libs/glfw/include/GLFW/glfw3.h
        libs/glfw/include/GLFW/glfw3native.h

        project/main.cpp
        project/uiFunctions/uiFunctions.h
        project/uiFunctions/uiFunctions.cpp
        project/simState/simState.cpp
        project/simState/simState.h project/simBuilder/simBuilder.cpp project/simBuilder/simBuilder.h project/utilities/utilities.h)

target_link_libraries(BachelorProject GL ${GLFW_STATIC_LIBRARIES})