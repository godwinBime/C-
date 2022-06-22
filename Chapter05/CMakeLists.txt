# CMakeList.txt : CMake project for Chpater05, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Chpater05")

set(CMAKE_CXX_STANDARD 11)

#IF(EXISTS ${CMAKE_BINARY_DIR}/ conanbuildinfo.cmake)
#	include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#	conan_basic_setup()
#ENDIF()

#Requires OpenCV
find_package(OpenCV 4.6.0 REQUIRED)
message("===> OpenCV Version: ${OpenCV_VERSION} ====>")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})


# Add source to this project's executable.
add_executable (${PROJECT_NAME} "utils/MultipleImageWindow.cpp" "utils/MultipleImageWindow.h" "main.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Chpater05 PROPERTY CXX_STANDARD 20)
endif()

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# TODO: Add tests and install targets if needed.
