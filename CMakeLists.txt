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

ADD_DEFINITIONS(-DUSE_KCF_TRACKING)
ADD_DEFINITIONS(-DUSE_CUDNN)
ADD_DEFINITIONS(-DUSE_OPENCV)
ADD_DEFINITIONS(-DDEBUG)
ADD_DEFINITIONS(-DLOOP_SEARCH)
#ADD_DEFINITIONS(-DLSH_SEARCH)
#ADD_DEFINITIONS(-DKDTREE_SEARCH)

SET(detour_SRCS
	src/util/util_config.cpp
	src/util/Point.cpp
	src/dataBase/dataBase.cpp
	src/kcf/fhog.cpp
	src/kcf/kcftracker.cpp
	src/faceAnalysis/faceAttri.cpp
	src/faceAnalysis/faceRecong.cpp
	src/faceAnalysis/objectbase.cpp
	src/faceAnalysis/objectDetect.cpp
	src/faceAnalysis/reId.cpp
	src/faceAnalysis/faceAnalysis.cpp
	src/faceAnalysis/reidInference.cpp
	src/sort/kalmanfilter.cpp
	src/sort/linear_assignment.cpp
	src/sort/model.cpp
	src/sort/nn_matching.cpp
	src/sort/track.cpp
	src/sort/tracker.cpp
	src/munkresAssignment/hungarianoper.cpp
	src/munkresAssignment/munkres.cpp
)
SET(detour_INCLUDE
	include/faceAnalysis
	include/lshbox
	include/kcf
	include/sort
	include/munkresAssignment
	include/eigen/Eigen
	include/eigen
	include/util
	include/dataBase
)

add_library(faceInference SHARED ${detour_SRCS} ${detour_INCLUDE})
SET(LIBRARY_OUTPUT_PATH ../lib)
target_link_libraries(faceInference ${OpenCV_LIBS} ${Caffe_LIB} ${GLOG_LIB} ${Boost_LIBRARIES} ${CUDA_LIB} -lpthread)

include_directories(include) 
FILE(GLOB_RECURSE sourcefiles "src/main.cpp")
add_executable( faceRecong ${sourcefiles} )
target_link_libraries( faceRecong faceInference)

FILE(GLOB_RECURSE testsourcefiles "src/testSample.cpp")
add_executable( test ${testsourcefiles} )
target_link_libraries( test faceInference)

FILE(GLOB_RECURSE sourcefiles "src/main_reid.cpp")
add_executable( reid_test ${sourcefiles} )
target_link_libraries( reid_test faceInference)

