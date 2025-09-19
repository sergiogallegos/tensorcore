cmake_minimum_required(VERSION 3.15)
project(TensorCore VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# Find BLAS
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "BLAS library not found")
endif()

# Find LAPACK
find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
else()
    message(FATAL_ERROR "LAPACK library not found")
endif()

# Find Python and pybind11 for Python bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(include)
include_directories(src)

# Source files
set(TENSORCORE_SOURCES
    src/tensor.cpp
    src/operations.cpp
    src/activations.cpp
    src/losses.cpp
    src/optimizers.cpp
    src/layers.cpp
    src/autograd.cpp
    src/blas_wrapper.cpp
    src/simd_utils.cpp
    src/memory_manager.cpp
    src/utils.cpp
)

# Header files
set(TENSORCORE_HEADERS
    include/tensorcore/tensor.hpp
    include/tensorcore/operations.hpp
    include/tensorcore/activations.hpp
    include/tensorcore/losses.hpp
    include/tensorcore/optimizers.hpp
    include/tensorcore/layers.hpp
    include/tensorcore/autograd.hpp
    include/tensorcore/utils.hpp
)

# Create the main library
add_library(tensorcore STATIC ${TENSORCORE_SOURCES} ${TENSORCORE_HEADERS})

# Link libraries
target_link_libraries(tensorcore 
    ${BLAS_LIBRARIES} 
    ${LAPACK_LIBRARIES}
)

# Set include directories for the library
target_include_directories(tensorcore PUBLIC include)

# Compiler definitions
target_compile_definitions(tensorcore PRIVATE
    TENSORCORE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    TENSORCORE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    TENSORCORE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Enable testing
enable_testing()

# Add subdirectories
add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(python)

# Installation
install(TARGETS tensorcore
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/tensorcore
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# CPack configuration
include(CPack)
set(CPACK_PACKAGE_NAME "TensorCore")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Educational Machine Learning Library")
set(CPACK_PACKAGE_VENDOR "TensorCore Contributors")

# Print configuration summary
message(STATUS "TensorCore Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "  LAPACK libraries: ${LAPACK_LIBRARIES}")
message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
message(STATUS "  pybind11 version: ${pybind11_VERSION}")
