cmake_minimum_required(VERSION 3.16)
project(LC VERSION 0.1.0)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(GNUInstallDirs)
find_package(OpenMP REQUIRED)
find_package(Python REQUIRED)

option(LC_FRAMEWORK_VERBOSE_GENERATOR "make the generator scripts run in verbose mode" OFF)
if(LC_FRAMEWORK_VERBOSE_GENERATOR)
    set(VERBOSE_FLAG "--verbose")
else()
    set(VERBOSE_FLAG "")
endif()

file(GLOB LC_SOURCES CONFIGURE_DEPENDS components/*.h verifiers/*.h preprocessors/*.h)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lc.cpp
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_Host_LC-Framework.py --output_dir ${CMAKE_CURRENT_BINARY_DIR} ${VERBOSE_FLAG}
    DEPENDS ${LC_SOURCES} generate_Host_LC-Framework.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_executable(lc ${CMAKE_CURRENT_BINARY_DIR}/lc.cpp)
target_include_directories(lc PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/components>
    )
target_compile_definitions(lc PRIVATE USE_CPU)
target_link_libraries(lc PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS lc)


