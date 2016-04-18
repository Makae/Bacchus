cmake_minimum_required(VERSION 3.0)
project(bacchus_project)
file(GLOB_RECURSE sources src/*.cpp include/project/*.h)

# add the data to the target, so it becomes visible in some IDE
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Bacchus ${sources})
target_link_libraries(Bacchus ${OpenCV_LIBS})