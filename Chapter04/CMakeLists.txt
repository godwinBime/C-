# CMakeList.txt : CMake project for Chapter04PhotoTool, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Chapter04PhotoTool")

set (CMAKE_CXX_STANDARD 11)

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
	include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
endif()

#Requires Opencv
find_package(OpenCV 4.6.0 REQUIRED)

message("===> OpenCV VERSION: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIRS})  

# Add source to this project's executable.
add_executable (${PROJECT_NAME} "Chapter04PhotoTool.cpp" "Chapter04PhotoTool.h")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Chapter04PhotoTool PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
