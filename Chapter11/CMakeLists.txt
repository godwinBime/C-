cmake_minimum_required(VERSION 3.22)
project(Chapter11)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

message("OpenCV Version::: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIB_DIRS})

add_executable(Chapter11 src/detection.cpp ocr.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
