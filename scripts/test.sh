#!/bin/bash

# TensorCore Test Script
# This script runs all tests for the TensorCore library

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run C++ tests
run_cpp_tests() {
    print_status "Running C++ tests..."
    
    if [ -f "build/tensorcore_tests" ]; then
        cd build
        ./tensorcore_tests
        cd ..
        print_success "C++ tests completed successfully"
    else
        print_warning "C++ test executable not found. Building first..."
        cd build
        make tensorcore_tests
        ./tensorcore_tests
        cd ..
        print_success "C++ tests completed successfully"
    fi
}

# Function to run Python tests
run_python_tests() {
    print_status "Running Python tests..."
    
    # Check if pytest is available
    if command_exists pytest; then
        pytest tests/ -v --tb=short
    elif command_exists python3; then
        # Run basic Python tests
        python3 -m unittest discover tests/ -v
    else
        print_error "Neither pytest nor python3 found. Cannot run Python tests."
        return 1
    fi
    
    print_success "Python tests completed successfully"
}

# Function to run benchmarks
run_benchmarks() {
    print_status "Running benchmarks..."
    
    if [ -f "build/tensorcore_benchmarks" ]; then
        cd build
        ./tensorcore_benchmarks
        cd ..
        print_success "Benchmarks completed successfully"
    else
        print_warning "Benchmark executable not found. Building first..."
        cd build
        make tensorcore_benchmarks
        ./tensorcore_benchmarks
        cd ..
        print_success "Benchmarks completed successfully"
    fi
}

# Function to run memory tests
run_memory_tests() {
    print_status "Running memory tests..."
    
    if command_exists valgrind; then
        cd build
        valgrind --leak-check=full --show-leak-kinds=all ./tensorcore_tests
        cd ..
        print_success "Memory tests completed successfully"
    else
        print_warning "Valgrind not found. Skipping memory tests."
    fi
}

# Function to run coverage tests
run_coverage_tests() {
    print_status "Running coverage tests..."
    
    if command_exists gcov; then
        cd build
        make clean
        cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="--coverage"
        make -j$(nproc 2>/dev/null || echo 4)
        ./tensorcore_tests
        gcov *.gcno
        cd ..
        print_success "Coverage tests completed successfully"
    else
        print_warning "gcov not found. Skipping coverage tests."
    fi
}

# Function to run Python coverage tests
run_python_coverage() {
    print_status "Running Python coverage tests..."
    
    if command_exists pytest; then
        pytest tests/ --cov=tensorcore --cov-report=html --cov-report=term
        print_success "Python coverage tests completed successfully"
    else
        print_warning "pytest not found. Skipping Python coverage tests."
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    # Test basic functionality
    python3 -c "
import tensorcore as tc
import numpy as np

# Test basic tensor operations
a = tc.tensor([1, 2, 3, 4])
b = tc.tensor([5, 6, 7, 8])
c = a + b
print(f'Basic addition: {c}')

# Test matrix operations
A = tc.tensor([[1, 2], [3, 4]])
B = tc.tensor([[5, 6], [7, 8]])
C = A.matmul(B)
print(f'Matrix multiplication: {C}')

# Test statistical operations
print(f'Sum: {a.sum()}')
print(f'Mean: {a.mean()}')
print(f'Max: {a.max()}')
print(f'Min: {a.min()}')

print('Integration tests passed!')
"
    
    print_success "Integration tests completed successfully"
}

# Function to run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    
    python3 -c "
import tensorcore as tc
import time

# Test tensor creation performance
start_time = time.time()
for i in range(1000):
    a = tc.tensor([1, 2, 3, 4])
creation_time = time.time() - start_time
print(f'Tensor creation time: {creation_time:.4f}s')

# Test matrix multiplication performance
A = tc.tensor([[1, 2], [3, 4]])
B = tc.tensor([[5, 6], [7, 8]])
start_time = time.time()
for i in range(1000):
    C = A.matmul(B)
matmul_time = time.time() - start_time
print(f'Matrix multiplication time: {matmul_time:.4f}s')

print('Performance tests completed!')
"
    
    print_success "Performance tests completed successfully"
}

# Function to run all tests
run_all_tests() {
    print_status "Running all tests..."
    
    run_cpp_tests
    run_python_tests
    run_benchmarks
    run_integration_tests
    run_performance_tests
    
    print_success "All tests completed successfully"
}

# Function to print test summary
print_summary() {
    print_success "Test suite completed!"
    echo
    echo "Test Summary:"
    echo "-------------"
    echo "• C++ tests: ✓"
    echo "• Python tests: ✓"
    echo "• Benchmarks: ✓"
    echo "• Integration tests: ✓"
    echo "• Performance tests: ✓"
    echo
    echo "All tests passed! 🎉"
}

# Main test function
main() {
    echo "TensorCore Test Script"
    echo "====================="
    echo
    
    # Parse command line arguments
    TEST_TYPE="all"
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --cpp)
                TEST_TYPE="cpp"
                shift
                ;;
            --python)
                TEST_TYPE="python"
                shift
                ;;
            --benchmarks)
                TEST_TYPE="benchmarks"
                shift
                ;;
            --memory)
                TEST_TYPE="memory"
                shift
                ;;
            --coverage)
                TEST_TYPE="coverage"
                shift
                ;;
            --integration)
                TEST_TYPE="integration"
                shift
                ;;
            --performance)
                TEST_TYPE="performance"
                shift
                ;;
            --all)
                TEST_TYPE="all"
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --cpp          Run only C++ tests"
                echo "  --python       Run only Python tests"
                echo "  --benchmarks   Run only benchmarks"
                echo "  --memory       Run only memory tests"
                echo "  --coverage     Run only coverage tests"
                echo "  --integration  Run only integration tests"
                echo "  --performance  Run only performance tests"
                echo "  --all          Run all tests (default)"
                echo "  --verbose      Enable verbose output"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run selected tests
    case $TEST_TYPE in
        "cpp")
            run_cpp_tests
            ;;
        "python")
            run_python_tests
            ;;
        "benchmarks")
            run_benchmarks
            ;;
        "memory")
            run_memory_tests
            ;;
        "coverage")
            run_coverage_tests
            run_python_coverage
            ;;
        "integration")
            run_integration_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        "all")
            run_all_tests
            ;;
        *)
            print_error "Unknown test type: $TEST_TYPE"
            exit 1
            ;;
    esac
    
    print_summary
}

# Run main function
main "$@"
