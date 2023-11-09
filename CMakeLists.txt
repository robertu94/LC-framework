cmake_minimum_required(VERSION 3.16)
project(LC VERSION 1.1.2)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
if(NOT USE_SANITIZER)
  set(USE_SANITIZER "" CACHE STRING "use a build sanitizer" FORCE)
endif()
set_property(CACHE USE_SANITIZER PROPERTY STRINGS "" "Address" "Thread" "Undefined" "Leak" "Memory")
if (USE_SANITIZER STREQUAL "Address")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
elseif (USE_SANITIZER STREQUAL "Thread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
elseif (USE_SANITIZER STREQUAL "Undefined")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
elseif (USE_SANITIZER STREQUAL "Leak")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
elseif (USE_SANITIZER STREQUAL "Memory")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(GNUInstallDirs)
find_package(OpenMP REQUIRED)
find_package(Python REQUIRED)

option(BUILD_SHARED_LIBS "prefer shared libraries" ON)
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
    DEPENDS ${LC_SOURCES} generate_Host_LC-Framework.py framework.cu
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_executable(lc ${CMAKE_CURRENT_BINARY_DIR}/lc.cpp)
target_include_directories(lc PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/components>
    )
target_compile_definitions(lc PRIVATE USE_CPU)
target_link_libraries(lc PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS lc EXPORT LCTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )


option(LC_BUILD_LIBPRESSIO_PLUGIN "build support for calling LC from LibPressio" OFF)
if(LC_BUILD_LIBPRESSIO_PLUGIN)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libpressio/)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libpressio/lc.cpp
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_Host_LC-Framework.py --base_file ${CMAKE_CURRENT_SOURCE_DIR}/lc-libpressio-plugin.cpp --output_dir ${CMAKE_CURRENT_BINARY_DIR}/libpressio/ ${VERBOSE_FLAG}
        DEPENDS ${LC_SOURCES} generate_Host_LC-Framework.py ./lc-libpressio-plugin.cpp
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    find_package(LibPressio REQUIRED)
    add_library(libpressio_lc ${CMAKE_CURRENT_BINARY_DIR}/libpressio/lc.cpp)
    target_link_libraries(libpressio_lc PRIVATE LibPressio::libpressio OpenMP::OpenMP_CXX)
    target_compile_definitions(libpressio_lc PRIVATE USE_CPU)
    target_include_directories(libpressio_lc PUBLIC 
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        )
    target_include_directories(libpressio_lc PRIVATE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/libpressio/components>
        )
    install(TARGETS libpressio_lc EXPORT LCTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    install(FILES lc_libpressio.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

install(EXPORT LCTargets NAMESPACE LC:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LC/)
export(EXPORT LCTargets FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/LCTargets.cmake NAMESPACE LC::)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/LCConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/LCConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LC
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LCConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LCConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LCConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LC
)


