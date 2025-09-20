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

# Set include directories for the library
target_include_directories(tensorcore PUBLIC include)

# Compiler definitions
target_compile_definitions(tensorcore PRIVATE
    TENSORCORE_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    TENSORCORE_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    TENSORCORE_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Create test executables
add_executable(test_basic src/test_basic.cpp)
target_link_libraries(test_basic tensorcore)

add_executable(test_tensor_operations tests/test_tensor_operations.cpp)
target_link_libraries(test_tensor_operations tensorcore)

add_executable(test_performance tests/test_performance.cpp)
target_link_libraries(test_performance tensorcore)

add_executable(test_edge_cases tests/test_edge_cases.cpp)
target_link_libraries(test_edge_cases tensorcore)

# Python bindings (optional)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
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
        python/tensorcore_core_minimal.cpp
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
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "  Python bindings: ON")
    message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
    message(STATUS "  pybind11 version: ${pybind11_VERSION}")
else()
    message(STATUS "  Python bindings: OFF")
endif()
