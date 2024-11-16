cmake_minimum_required(VERSION 3.16)
project(LC VERSION 1.3.0)

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
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cpu)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cpu/lc.h ${CMAKE_CURRENT_BINARY_DIR}/cpu/lc.cpp ${CMAKE_CURRENT_BINARY_DIR}/cpu/include/consts.h
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_Host_LC-Framework.py --output_dir ${CMAKE_CURRENT_BINARY_DIR}/cpu ${VERBOSE_FLAG}
    DEPENDS ${LC_SOURCES} generate_Host_LC-Framework.py framework.cu
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_executable(lc ${CMAKE_CURRENT_BINARY_DIR}/cpu/lc.cpp)
target_include_directories(lc PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/cpu/components>
    )
target_compile_definitions(lc PRIVATE USE_CPU)
target_link_libraries(lc PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS lc EXPORT LCTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT CPU
    )

option(LC_BUILD_CUDA "enable support for cuda GPUs" OFF)
if(LC_BUILD_CUDA)
enable_language(CUDA)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gpu)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpu/lc.h ${CMAKE_CURRENT_BINARY_DIR}/gpu/lc.cu ${CMAKE_CURRENT_BINARY_DIR}/gpu/include/consts.h
        COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_Device_LC-Framework.py --output_dir ${CMAKE_CURRENT_BINARY_DIR}/gpu ${VERBOSE_FLAG}
        DEPENDS ${LC_SOURCES} generate_Device_LC-Framework.py framework.cu
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    add_executable(lc-gpu ${CMAKE_CURRENT_BINARY_DIR}/gpu/lc.cu)
    target_include_directories(lc-gpu PRIVATE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gpu/components>
        )
    target_compile_definitions(lc-gpu PRIVATE USE_GPU)
    target_link_libraries(lc-gpu PRIVATE OpenMP::OpenMP_CXX)
    install(TARGETS lc-gpu EXPORT LCTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT GPU
        )
endif()



option(LC_BUILD_LIBPRESSIO_PLUGIN "build support for calling LC from LibPressio" OFF)
if(LC_BUILD_LIBPRESSIO_PLUGIN)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libpressio/)
    find_package(LibPressio REQUIRED)
    add_library(libpressio_lc 
        lc-libpressio-plugin.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/cpu/include/consts.h
        )

    add_dependencies(libpressio_lc lc)
    target_link_libraries(libpressio_lc PRIVATE LibPressio::libpressio OpenMP::OpenMP_CXX)
    target_compile_definitions(libpressio_lc PRIVATE USE_CPU)
    target_include_directories(libpressio_lc PUBLIC 
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
        )
    target_include_directories(libpressio_lc PRIVATE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/cpu/>
        )
    install(TARGETS libpressio_lc EXPORT LCTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT LIBPRESSIO
        )
    install(FILES lc_libpressio.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT LIBPRESSIO)

    if(LC_BUILD_CUDA)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/libpressio_gpu/)
        file(COPY_FILE lc-libpressio-plugin.cpp ${CMAKE_CURRENT_BINARY_DIR}/lc-libpressio-plugin-gpu.cu)
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lc-libpressio-plugin-gpu.cu
            COMMAND cmake -E copy_if_different lc-libpressio-plugin.cpp ${CMAKE_CURRENT_BINARY_DIR}/lc-libpressio-plugin-gpu.cu
            DEPENDS lc-libpressio-plugin.cpp
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_library(libpressio_lc_gpu
            ${CMAKE_CURRENT_BINARY_DIR}/lc-libpressio-plugin-gpu.cu
            ${CMAKE_CURRENT_BINARY_DIR}/gpu/lc.h
            ${CMAKE_CURRENT_BINARY_DIR}/gpu/include/consts.h
            )
        add_dependencies(libpressio_lc_gpu lc-gpu)
        target_link_libraries(libpressio_lc_gpu PRIVATE LibPressio::libpressio OpenMP::OpenMP_CXX)
        target_compile_definitions(libpressio_lc_gpu PRIVATE USE_GPU)
        target_include_directories(libpressio_lc_gpu PRIVATE 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gpu/>
            )
        target_include_directories(libpressio_lc_gpu PUBLIC 
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
            )
        install(TARGETS libpressio_lc_gpu EXPORT LCTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT LIBPRESSIO_GPU
            )
        install(FILES lc_gpu_libpressio.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT LIBPRESSIO_GPU)
    endif()
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


