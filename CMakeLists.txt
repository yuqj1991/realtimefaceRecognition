cmake_minimum_required(VERSION 2.8)
project(FaceTracking)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3  -march=native -std=c++11 -fopenmp")
set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem regex  REQUIRED)


set(Caffe_LIB caffe)
include_directories(include)

link_directories(../lib)

set(GLOG_LIB glog)



find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})


#ADD_DEFINITIONS(-DCPU_ONLY=0)
ADD_DEFINITIONS(-DUSE_CUDNN=1)
ADD_DEFINITIONS(-DDEBUG=1)
ADD_DEFINITIONS(-DLOOP_SEARCH=1)
#ADD_DEFINITIONS(-DLSH_SEARCH=1)
#ADD_DEFINITIONS(-DKDTREE_SEARCH=1)


SET(detour_SRCS
	src/Point.cpp
	src/dataBase.cpp
	src/kcf/fhog.cpp
	src/kcf/kcftracker.cpp
	src/faceAnalysis/faceAttri.cpp
	src/faceAnalysis/faceRecong.cpp
	src/faceAnalysis/objectbase.cpp
	src/faceAnalysis/objectDetect.cpp
	src/faceAnalysis/reId.cpp
	src/faceAnalysis/faceAnalysis.cpp
)
SET(detour_HDRS
	include/Point.hpp
	include/utils_config.hpp
	include/dataBase.hpp
	include/kdtree.hpp
	include/lshbox.h
	include/faceAnalysis
	include/eigen
	include/lshbox
	include/kcf
)

add_library(faceInference SHARED ${detour_SRCS} ${detour_HDRS})
SET(LIBRARY_OUTPUT_PATH ../lib)
target_link_libraries(faceInference ${OpenCV_LIBS} ${Caffe_LIB} ${GLOG_LIB} ${Boost_LIBRARIES} ${CUDA_LIB})

include_directories(include) 
FILE(GLOB_RECURSE sourcefiles "src/main.cpp")
add_executable( faceRecong ${sourcefiles} )
target_link_libraries( faceRecong faceInference)

FILE(GLOB_RECURSE testsourcefiles "src/testSample.cpp")
add_executable( test ${testsourcefiles} )
target_link_libraries( test faceInference)

