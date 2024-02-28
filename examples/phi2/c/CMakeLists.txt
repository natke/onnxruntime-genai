cmake_minimum_required(VERSION 3.26)
project(phi2)

set(ORT_HOME C:\\Users\\nakersha\\Develop\\install\\onnxruntime)
set(ORT_GENAI_HOME onnxruntime-genai)


###############################################################################
## file globbing ##############################################################
###############################################################################

# these instructions search the directory tree when CMake is
# invoked and put all files that match the pattern in the variables
# `sources` and `data`
file(GLOB_RECURSE sources      phi.c)
#file(GLOB_RECURSE sources_test src/test/*.cpp)
#file(GLOB_RECURSE data resources/*)
# you can use set(sources src/main.cpp) etc if you don't want to
# use globbing to find files automatically

###############################################################################
## target definitions #########################################################
###############################################################################

# add the data to the target, so it becomes visible in some IDE
add_executable(phi2 ${sources})

# just for phi2 add some compiler flags
#target_compile_options(phi2 PUBLIC -std=c++11 -Wall -pedantic -Wextra)

# this lets me include files relative to the root source directory with a <> pair
target_include_directories(phi2 PUBLIC ${ORT_HOME}/include ${ORT_GENAI_HOME}/include)

# this copies all resource files in the build directory
# we need this, because we want to work with paths relative to the executable
#file(COPY ${data} DESTINATION resources)

###############################################################################
## dependencies ###############################################################
###############################################################################

# this defines the variables Boost_LIBRARIES that contain all library names
# that we need to link to
#find_package(Boost 1.36.0 COMPONENTS filesystem system REQUIRED)

target_link_directories(phi2 PUBLIC ${ORT_GENAI_HOME}/lib ${ORT_HOME}/lib )

target_link_libraries(phi2 PUBLIC
  onnxruntime-genai
  onnxruntime
  # here you can add any library dependencies
)