# Test configuration
find_package(GTest REQUIRED)

# Test source files
set(TEST_SOURCES
    test_tensor.cpp
    test_operations.cpp
    test_activations.cpp
    test_losses.cpp
    test_optimizers.cpp
    test_layers.cpp
    test_autograd.cpp
    test_utils.cpp
)

# Create test executable
add_executable(tensorcore_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(tensorcore_tests
    tensorcore
    GTest::gtest
    GTest::gtest_main
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Include directories
target_include_directories(tensorcore_tests PRIVATE include)

# Add tests to CTest
add_test(NAME tensorcore_tests COMMAND tensorcore_tests)

# Set test properties
set_tests_properties(tensorcore_tests PROPERTIES
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
