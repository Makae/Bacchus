cmake_minimum_required(VERSION 3.0)
project(bacchus_project)
file(GLOB_RECURSE sources src/*.cpp include/project/*.h)

# add the data to the target, so it becomes visible in some IDE
add_executable(Bacchus ${sources})
set(OpenCV_DIR $ENV{OPENCV_DIR}\\..\\..\\)
MESSAGE($ENV{OPENCV_DIR}\\lib)
find_package(OpenCV)
target_link_libraries(Bacchus ${OpenCV_LIBS})