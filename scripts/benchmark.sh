#!/bin/bash

# TensorCore Benchmark Script
# This script runs performance benchmarks for the TensorCore library

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

# Function to run tensor operations benchmark
run_tensor_benchmark() {
    print_status "Running tensor operations benchmark..."
    
    python3 -c "
import tensorcore as tc
import time
import statistics

def benchmark_tensor_operations():
    print('Tensor Operations Benchmark')
    print('=' * 40)
    
    # Test tensor creation
    times = []
    for i in range(1000):
        start = time.time()
        a = tc.tensor([1, 2, 3, 4, 5])
        end = time.time()
        times.append(end - start)
    
    print(f'Tensor creation: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test tensor addition
    a = tc.tensor([1, 2, 3, 4, 5])
    b = tc.tensor([6, 7, 8, 9, 10])
    
    times = []
    for i in range(1000):
        start = time.time()
        c = a + b
        end = time.time()
        times.append(end - start)
    
    print(f'Tensor addition: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test tensor multiplication
    times = []
    for i in range(1000):
        start = time.time()
        c = a * b
        end = time.time()
        times.append(end - start)
    
    print(f'Tensor multiplication: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test tensor sum
    times = []
    for i in range(1000):
        start = time.time()
        s = a.sum()
        end = time.time()
        times.append(end - start)
    
    print(f'Tensor sum: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')

benchmark_tensor_operations()
"
    
    print_success "Tensor operations benchmark completed"
}

# Function to run matrix operations benchmark
run_matrix_benchmark() {
    print_status "Running matrix operations benchmark..."
    
    python3 -c "
import tensorcore as tc
import time
import statistics

def benchmark_matrix_operations():
    print('Matrix Operations Benchmark')
    print('=' * 40)
    
    # Test matrix creation
    times = []
    for i in range(1000):
        start = time.time()
        A = tc.tensor([[1, 2], [3, 4]])
        end = time.time()
        times.append(end - start)
    
    print(f'Matrix creation: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test matrix multiplication
    A = tc.tensor([[1, 2], [3, 4]])
    B = tc.tensor([[5, 6], [7, 8]])
    
    times = []
    for i in range(1000):
        start = time.time()
        C = A.matmul(B)
        end = time.time()
        times.append(end - start)
    
    print(f'Matrix multiplication: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test matrix transpose
    times = []
    for i in range(1000):
        start = time.time()
        T = A.transpose()
        end = time.time()
        times.append(end - start)
    
    print(f'Matrix transpose: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test matrix determinant
    times = []
    for i in range(1000):
        start = time.time()
        d = A.det()
        end = time.time()
        times.append(end - start)
    
    print(f'Matrix determinant: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')

benchmark_matrix_operations()
"
    
    print_success "Matrix operations benchmark completed"
}

# Function to run neural network benchmark
run_nn_benchmark() {
    print_status "Running neural network benchmark..."
    
    python3 -c "
import tensorcore as tc
import time
import statistics

def benchmark_neural_network():
    print('Neural Network Benchmark')
    print('=' * 40)
    
    # Test dense layer forward pass
    from tensorcore.nn import Dense
    
    layer = Dense(784, 128)
    x = tc.tensor.random_normal((32, 784))
    
    times = []
    for i in range(1000):
        start = time.time()
        y = layer.forward(x)
        end = time.time()
        times.append(end - start)
    
    print(f'Dense layer forward: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test activation functions
    from tensorcore.nn import ReLU, Sigmoid, Tanh
    
    x = tc.tensor.random_normal((32, 128))
    
    # ReLU
    times = []
    for i in range(1000):
        start = time.time()
        y = ReLU.forward(x)
        end = time.time()
        times.append(end - start)
    
    print(f'ReLU activation: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Sigmoid
    times = []
    for i in range(1000):
        start = time.time()
        y = Sigmoid.forward(x)
        end = time.time()
        times.append(end - start)
    
    print(f'Sigmoid activation: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Tanh
    times = []
    for i in range(1000):
        start = time.time()
        y = Tanh.forward(x)
        end = time.time()
        times.append(end - start)
    
    print(f'Tanh activation: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')

benchmark_neural_network()
"
    
    print_success "Neural network benchmark completed"
}

# Function to run optimization benchmark
run_optimizer_benchmark() {
    print_status "Running optimizer benchmark..."
    
    python3 -c "
import tensorcore as tc
import time
import statistics

def benchmark_optimizers():
    print('Optimizer Benchmark')
    print('=' * 40)
    
    from tensorcore.optim import SGD, Adam, RMSprop
    
    # Create test parameters
    params = [tc.tensor.random_normal((100, 100)) for _ in range(10)]
    
    # Test SGD
    sgd = SGD(params, lr=0.01)
    times = []
    for i in range(1000):
        start = time.time()
        sgd.step()
        end = time.time()
        times.append(end - start)
    
    print(f'SGD step: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test Adam
    adam = Adam(params, lr=0.001)
    times = []
    for i in range(1000):
        start = time.time()
        adam.step()
        end = time.time()
        times.append(end - start)
    
    print(f'Adam step: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')
    
    # Test RMSprop
    rmsprop = RMSprop(params, lr=0.01)
    times = []
    for i in range(1000):
        start = time.time()
        rmsprop.step()
        end = time.time()
        times.append(end - start)
    
    print(f'RMSprop step: {statistics.mean(times)*1000:.4f}ms ± {statistics.stdev(times)*1000:.4f}ms')

benchmark_optimizers()
"
    
    print_success "Optimizer benchmark completed"
}

# Function to run memory benchmark
run_memory_benchmark() {
    print_status "Running memory benchmark..."
    
    python3 -c "
import tensorcore as tc
import time
import psutil
import os

def benchmark_memory():
    print('Memory Benchmark')
    print('=' * 40)
    
    process = psutil.Process(os.getpid())
    
    # Test memory usage for large tensors
    sizes = [100, 500, 1000, 2000, 5000]
    
    for size in sizes:
        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large tensor
        start = time.time()
        tensor = tc.tensor.random_normal((size, size))
        creation_time = time.time() - start
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f'Size {size}x{size}: {creation_time:.4f}s, {memory_used:.2f}MB')
        
        # Clean up
        del tensor

benchmark_memory()
"
    
    print_success "Memory benchmark completed"
}

# Function to run C++ benchmarks
run_cpp_benchmarks() {
    print_status "Running C++ benchmarks..."
    
    if [ -f "build/tensorcore_benchmarks" ]; then
        cd build
        ./tensorcore_benchmarks
        cd ..
        print_success "C++ benchmarks completed"
    else
        print_warning "C++ benchmark executable not found. Building first..."
        cd build
        make tensorcore_benchmarks
        ./tensorcore_benchmarks
        cd ..
        print_success "C++ benchmarks completed"
    fi
}

# Function to compare with NumPy
compare_with_numpy() {
    print_status "Comparing with NumPy..."
    
    python3 -c "
import tensorcore as tc
import numpy as np
import time
import statistics

def compare_operations():
    print('TensorCore vs NumPy Comparison')
    print('=' * 40)
    
    # Test data
    size = 1000
    a_tc = tc.tensor.random_normal((size, size))
    b_tc = tc.tensor.random_normal((size, size))
    
    a_np = np.random.normal(0, 1, (size, size))
    b_np = np.random.normal(0, 1, (size, size))
    
    # Matrix multiplication comparison
    times_tc = []
    for i in range(100):
        start = time.time()
        c_tc = a_tc.matmul(b_tc)
        end = time.time()
        times_tc.append(end - start)
    
    times_np = []
    for i in range(100):
        start = time.time()
        c_np = np.matmul(a_np, b_np)
        end = time.time()
        times_np.append(end - start)
    
    print(f'Matrix multiplication ({size}x{size}):')
    print(f'  TensorCore: {statistics.mean(times_tc)*1000:.4f}ms ± {statistics.stdev(times_tc)*1000:.4f}ms')
    print(f'  NumPy:      {statistics.mean(times_np)*1000:.4f}ms ± {statistics.stdev(times_np)*1000:.4f}ms')
    print(f'  Speedup:    {statistics.mean(times_np)/statistics.mean(times_tc):.2f}x')
    
    # Element-wise operations comparison
    times_tc = []
    for i in range(1000):
        start = time.time()
        c_tc = a_tc + b_tc
        end = time.time()
        times_tc.append(end - start)
    
    times_np = []
    for i in range(1000):
        start = time.time()
        c_np = a_np + b_np
        end = time.time()
        times_np.append(end - start)
    
    print(f'Element-wise addition:')
    print(f'  TensorCore: {statistics.mean(times_tc)*1000:.4f}ms ± {statistics.stdev(times_tc)*1000:.4f}ms')
    print(f'  NumPy:      {statistics.mean(times_np)*1000:.4f}ms ± {statistics.stdev(times_np)*1000:.4f}ms')
    print(f'  Speedup:    {statistics.mean(times_np)/statistics.mean(times_tc):.2f}x')

compare_operations()
"
    
    print_success "NumPy comparison completed"
}

# Function to run all benchmarks
run_all_benchmarks() {
    print_status "Running all benchmarks..."
    
    run_tensor_benchmark
    run_matrix_benchmark
    run_nn_benchmark
    run_optimizer_benchmark
    run_memory_benchmark
    run_cpp_benchmarks
    compare_with_numpy
    
    print_success "All benchmarks completed"
}

# Function to print benchmark summary
print_summary() {
    print_success "Benchmark suite completed!"
    echo
    echo "Benchmark Summary:"
    echo "------------------"
    echo "• Tensor operations: ✓"
    echo "• Matrix operations: ✓"
    echo "• Neural network: ✓"
    echo "• Optimizers: ✓"
    echo "• Memory usage: ✓"
    echo "• C++ benchmarks: ✓"
    echo "• NumPy comparison: ✓"
    echo
    echo "All benchmarks completed! 📊"
}

# Main benchmark function
main() {
    echo "TensorCore Benchmark Script"
    echo "=========================="
    echo
    
    # Parse command line arguments
    BENCHMARK_TYPE="all"
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tensor)
                BENCHMARK_TYPE="tensor"
                shift
                ;;
            --matrix)
                BENCHMARK_TYPE="matrix"
                shift
                ;;
            --nn)
                BENCHMARK_TYPE="nn"
                shift
                ;;
            --optimizer)
                BENCHMARK_TYPE="optimizer"
                shift
                ;;
            --memory)
                BENCHMARK_TYPE="memory"
                shift
                ;;
            --cpp)
                BENCHMARK_TYPE="cpp"
                shift
                ;;
            --numpy)
                BENCHMARK_TYPE="numpy"
                shift
                ;;
            --all)
                BENCHMARK_TYPE="all"
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
                echo "  --tensor      Run only tensor operation benchmarks"
                echo "  --matrix      Run only matrix operation benchmarks"
                echo "  --nn          Run only neural network benchmarks"
                echo "  --optimizer   Run only optimizer benchmarks"
                echo "  --memory      Run only memory benchmarks"
                echo "  --cpp         Run only C++ benchmarks"
                echo "  --numpy       Run only NumPy comparison"
                echo "  --all         Run all benchmarks (default)"
                echo "  --verbose     Enable verbose output"
                echo "  --help        Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run selected benchmarks
    case $BENCHMARK_TYPE in
        "tensor")
            run_tensor_benchmark
            ;;
        "matrix")
            run_matrix_benchmark
            ;;
        "nn")
            run_nn_benchmark
            ;;
        "optimizer")
            run_optimizer_benchmark
            ;;
        "memory")
            run_memory_benchmark
            ;;
        "cpp")
            run_cpp_benchmarks
            ;;
        "numpy")
            compare_with_numpy
            ;;
        "all")
            run_all_benchmarks
            ;;
        *)
            print_error "Unknown benchmark type: $BENCHMARK_TYPE"
            exit 1
            ;;
    esac
    
    print_summary
}

# Run main function
main "$@"
