cmake_minimum_required(VERSION 3.15)
project(TensorCore VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(TENSORCORE_BUILD_EXPERIMENTAL "Build experimental ML, autograd, optimizer, sklearn, and memory-pool modules" OFF)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(include)
include_directories(src)

# Source files
set(TENSORCORE_SOURCES
    src/tensor.cpp
    src/operations.cpp
    src/activations.cpp
    src/losses.cpp
    src/utils.cpp
    src/simd_utils.cpp
)

# Header files
set(TENSORCORE_HEADERS
    include/tensorcore/tensor.hpp
    include/tensorcore/operations.hpp
    include/tensorcore/activations.hpp
    include/tensorcore/losses.hpp
    include/tensorcore/utils.hpp
    include/tensorcore/simd_utils.hpp
)

if(TENSORCORE_BUILD_EXPERIMENTAL)
    list(APPEND TENSORCORE_SOURCES
        src/optimizers.cpp
        src/layers.cpp
        src/autograd.cpp
        src/sklearn.cpp
        src/memory_pool.cpp
    )

    list(APPEND TENSORCORE_HEADERS
        include/tensorcore/optimizers.hpp
        include/tensorcore/layers.hpp
        include/tensorcore/autograd.hpp
        include/tensorcore/sklearn.hpp
        include/tensorcore/memory_pool.hpp
    )
endif()

# Create the main library
add_library(tensorcore STATIC ${TENSORCORE_SOURCES} ${TENSORCORE_HEADERS})

# Set include directories for the library
target_include_directories(tensorcore PUBLIC include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/simd_utils.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

# Compiler definitions
target_compile_definitions(tensorcore PRIVATE
    TENSORCORE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    TENSORCORE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    TENSORCORE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    TENSORCORE_BUILDING_LIBRARY=1
)

if(TENSORCORE_BUILD_EXPERIMENTAL)
    target_compile_definitions(tensorcore PUBLIC TENSORCORE_EXPERIMENTAL_API=1)
endif()

include(CTest)

# Create test executables
add_executable(test_basic tests/test_basic.cpp)
target_link_libraries(test_basic tensorcore)
add_test(NAME test_basic COMMAND test_basic)

add_executable(test_tensor_operations tests/test_tensor_operations.cpp)
target_link_libraries(test_tensor_operations tensorcore)
add_test(NAME test_tensor_operations COMMAND test_tensor_operations)

add_executable(test_performance tests/test_performance.cpp)
target_link_libraries(test_performance tensorcore)

add_executable(test_edge_cases tests/test_edge_cases.cpp)
target_link_libraries(test_edge_cases tensorcore)
add_test(NAME test_edge_cases COMMAND test_edge_cases)

# Python bindings (optional)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    if(NOT TENSORCORE_BUILD_EXPERIMENTAL)
        message(FATAL_ERROR "Python bindings currently expose experimental APIs. Configure with -DTENSORCORE_BUILD_EXPERIMENTAL=ON to build them.")
    endif()

    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    
    # Find pybind11 headers
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
    
    # Create Python module
    pybind11_add_module(tensorcore_core
        python/tensorcore_core.cpp
    )
    
    target_link_libraries(tensorcore_core PRIVATE tensorcore)
    target_include_directories(tensorcore_core PRIVATE include)
    
    # Set output directory for Python module
    set_target_properties(tensorcore_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
endif()

# Print configuration summary
message(STATUS "TensorCore Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Experimental modules: ${TENSORCORE_BUILD_EXPERIMENTAL}")
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "  Python bindings: ON")
    message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "  pybind11 version: ${pybind11_VERSION}")
else()
    message(STATUS "  Python bindings: OFF")
endif()
