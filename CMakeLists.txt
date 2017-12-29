cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project( nonoverlap )
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   -msse3 -pg  -std=c++11 -Wall -g -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse3 -pg  -std=c++11 -Wall -O3    ")
SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(aruco REQUIRED )
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable( boardM2TbyList calibrateM2TbyList_board.cpp )
add_executable( boardT2TbyList calibrateT2TbyList_board.cpp )
add_executable( synctestboardbyList SyntestM2T_boardbyList.cpp )
add_executable( synctestT2TbyList SynctestT2TbyList_board.cpp )

target_link_libraries( boardM2TbyList ${OpenCV_LIBS} ${aruco_LIBS} ${PCL_LIBRARIES})
target_link_libraries( boardT2TbyList ${OpenCV_LIBS} ${aruco_LIBS} ${PCL_LIBRARIES})
target_link_libraries( synctestboardbyList ${OpenCV_LIBS} ${aruco_LIBS} ${PCL_LIBRARIES})
target_link_libraries( synctestT2TbyList ${OpenCV_LIBS} ${aruco_LIBS} ${PCL_LIBRARIES})

