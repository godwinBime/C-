# CMakeList.txt : CMake project for CMakeConfigDemo, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

cmake_policy(SET CMP0012 NEW)

project ("CMakeConfigDemo")

#set(OpenCV_DIR "C://C++ EXTERNAL LIBS//opencv" )

# Add source to this project's executable.
add_executable (CMakeConfigDemo "CMakeConfigDemo.cpp" "CMakeConfigDemo.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeConfigDemo PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

find_package(OpenCV REQUIRED PATHS "C://C++ EXTERNAL LIBS//opencv//build//x64//vc15//lib" )

message("OpenCV VERSION: ${OpenCV_VERSION}")


include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIB_DIR})

target_link_libraries(CMakeConfigDemo ${OpenCV_LIBS})

#Set execution directory
#set_target_properties(${PROJECT_NAME} PROPERTIES VS_USER_PROPS "${PROJECT_NAME}.cpp.user.props")
