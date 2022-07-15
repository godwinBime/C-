cmake_minimum_required(VERSION 3.22)

project(Chapter08)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

message(OpenCV  Version::: ${OpenCV_VERSION})

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIB_DIRS})

add_executable(Chapter08 main/frameDifferencing.cpp main/backgroundSubtraction.cpp main/erosion.cpp main/morphologicalOperations.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
