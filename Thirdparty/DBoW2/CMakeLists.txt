cmake_minimum_required(VERSION 2.8)
project(DBoW2)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Compiler specific options for gcc
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native") 
# SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native") 

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(HDRS_DBOW2
    DBoW2/BowVector.h
    DBoW2/FORB.h
    DBoW2/FClass.h
    DBoW2/FeatureVector.h
    DBoW2/ScoringObject.h
    DBoW2/TemplatedVocabulary.h)
set(SRCS_DBOW2
    DBoW2/BowVector.cpp
    DBoW2/FORB.cpp
    DBoW2/FeatureVector.cpp
    DBoW2/ScoringObject.cpp)

set(HDRS_DUTILS
    DUtils/Random.h
    DUtils/Timestamp.h)
set(SRCS_DUTILS
    DUtils/Random.cpp
    DUtils/Timestamp.cpp)

find_package(OpenCV 4)
if(OpenCV_FOUND)
    add_definitions(-DOPENCV4)
else()
    find_package(OpenCV 3)
    if(NOT OpenCV_FOUND)
        find_package(OpenCV 2.4.3 QUIET)
        if(NOT OpenCV_FOUND)
            message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
        endif()
    endif()
endif()
message(STATUS "OpenCV Version: ${OpenCV_VERSION}.")

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
add_library(DBoW2 SHARED ${SRCS_DBOW2} ${SRCS_DUTILS})
target_link_libraries(DBoW2 ${OpenCV_LIBS})

