cmake_minimum_required(VERSION 3.0)
project(Bacchus)
file(GLOB_RECURSE headers include/project/*.h)
file(GLOB_RECURSE sources src/*.cpp)

# add the data to the target, so it becomes visible in some IDE
set(raspicam_DIR "/mnt/data/libs/raspicam/build")
find_package(OpenCV REQUIRED)
find_package(raspicam REQUIRED)
find_package(flandmark REQUIRED)
IF(flandmark_FOUND)
	MESSAGE(STATUS "GOOD, flandmark found")
	MESSAGE(STATUS ${flandmark_LIBS})
ELSEIF()
	MESSAGE(STATUS "BAD, no flandmark")
ENDIF()



IF(raspicam_FOUND)
	MESSAGE(STATUS "GOOD, raspicam found")
	MESSAGE(STATUS ${raspicam_LIBS})
ELSEIF()
	MESSAGE(STATUS "BAD, no raspicam")
ENDIF()


include_directories(${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "HDRS: ${headers}")
MESSAGE(STATUS "SRCS: ${sources}")
add_executable(Bacchus ${headers} ${sources})
target_link_libraries(Bacchus ${OpenCV_LIBS})

IF(raspicam_FOUND)
	target_link_libraries(Bacchus ${raspicam_LIBS})
	target_link_libraries(Bacchus ${raspicam_CV_LIBS})
ENDIF()

IF(flandmark_FOUND)
	target_link_libraries(Bacchus ${flandmark_LIBS})
ENDIF()

set(GCC_COMPILER_VERSION "-std=c++0x")
add_definitions(${GCC_COMPILER_VERSION})
