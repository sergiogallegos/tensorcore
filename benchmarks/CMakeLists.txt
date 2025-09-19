# Benchmark configuration
find_package(benchmark REQUIRED)

# Benchmark source files
set(BENCHMARK_SOURCES
    tensor_benchmarks.cpp
    operations_benchmarks.cpp
    activations_benchmarks.cpp
    losses_benchmarks.cpp
)

# Create benchmark executable
add_executable(tensorcore_benchmarks ${BENCHMARK_SOURCES})

# Link libraries
target_link_libraries(tensorcore_benchmarks
    tensorcore
    benchmark::benchmark
    benchmark::benchmark_main
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Include directories
target_include_directories(tensorcore_benchmarks PRIVATE include)

# Compiler definitions for benchmarks
target_compile_definitions(tensorcore_benchmarks PRIVATE
    BENCHMARK_ENABLE_TESTING=1
)
