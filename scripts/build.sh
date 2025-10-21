#!/bin/bash

# TensorCore Build Script
# This script builds the TensorCore library and Python bindings

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

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check for CMake
    if ! command_exists cmake; then
        print_error "CMake is not installed. Please install CMake 3.15 or later."
        exit 1
    fi
    
    # Check CMake version
    CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
    print_status "Found CMake version: $CMAKE_VERSION"
    
    # Check for Python
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.8 or later."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Found Python version: $PYTHON_VERSION"
    
    # Check for pip
    if ! command_exists pip3; then
        print_error "pip3 is not installed. Please install pip3."
        exit 1
    fi
    
    # Check for BLAS library
    if ! pkg-config --exists blas; then
        print_warning "BLAS library not found via pkg-config. Make sure BLAS is installed."
    else
        print_status "Found BLAS library"
    fi
    
    # Check for LAPACK library
    if ! pkg-config --exists lapack; then
        print_warning "LAPACK library not found via pkg-config. Make sure LAPACK is installed."
    else
        print_status "Found LAPACK library"
    fi
    
    print_success "Dependencies check completed"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install pybind11
    pip3 install pybind11
    
    # Install other dependencies
    pip3 install numpy scipy matplotlib pandas scikit-learn
    
    print_success "Python dependencies installed"
}

# Function to create build directory
create_build_dir() {
    print_status "Creating build directory..."
    
    if [ -d "build" ]; then
        print_warning "Build directory already exists. Cleaning..."
        rm -rf build
    fi
    
    mkdir -p build
    cd build
    
    print_success "Build directory created"
}

# Function to configure with CMake
configure_cmake() {
    print_status "Configuring with CMake..."
    
    # Get the number of CPU cores
    if command_exists nproc; then
        CORES=$(nproc)
    elif command_exists sysctl; then
        CORES=$(sysctl -n hw.ncpu)
    else
        CORES=4
    fi
    
    print_status "Using $CORES cores for compilation"
    
    # Configure CMake
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_CXX_EXTENSIONS=OFF \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    print_success "CMake configuration completed"
}

# Function to build the library
build_library() {
    print_status "Building TensorCore library..."
    
    # Build with all available cores
    make -j$(nproc 2>/dev/null || echo 4)
    
    print_success "TensorCore library built successfully"
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if [ -f "tensorcore_tests" ]; then
        ./tensorcore_tests
        print_success "Tests completed successfully"
    else
        print_warning "Test executable not found. Skipping tests."
    fi
}

# Function to install the library
install_library() {
    print_status "Installing TensorCore library..."
    
    make install
    
    print_success "TensorCore library installed"
}

# Function to build Python bindings
build_python_bindings() {
    print_status "Building Python bindings..."
    
    cd ../python
    
    # Install in development mode
    pip3 install -e .
    
    print_success "Python bindings built and installed"
}

# Function to run Python tests
run_python_tests() {
    print_status "Running Python tests..."
    
    if [ -f "test_python.py" ]; then
        python3 test_python.py
        print_success "Python tests completed successfully"
    else
        print_warning "Python test file not found. Skipping Python tests."
    fi
}

# Function to create examples
create_examples() {
    print_status "Creating example files..."
    
    cd ..
    
    # Create basic example
    cat > examples/basic_example.py << 'EOF'
#!/usr/bin/env python3
"""
Basic TensorCore example
"""

import tensorcore as tc

def main():
    print("TensorCore Basic Example")
    print("=" * 30)
    
    # Create tensors
    a = tc.tensor([1, 2, 3, 4])
    b = tc.tensor([5, 6, 7, 8])
    
    print(f"Tensor a: {a}")
    print(f"Tensor b: {b}")
    
    # Basic operations
    c = a + b
    print(f"a + b = {c}")
    
    d = a * b
    print(f"a * b = {d}")
    
    # Matrix operations
    A = tc.tensor([[1, 2], [3, 4]])
    B = tc.tensor([[5, 6], [7, 8]])
    
    print(f"\nMatrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    C = A.matmul(B)
    print(f"A @ B = \n{C}")
    
    # Statistical operations
    print(f"\nSum of a: {a.sum()}")
    print(f"Mean of a: {a.mean()}")
    print(f"Max of a: {a.max()}")
    print(f"Min of a: {a.min()}")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x examples/basic_example.py
    
    print_success "Example files created"
}

# Function to print build summary
print_summary() {
    print_success "Build completed successfully!"
    echo
    echo "Summary:"
    echo "--------"
    echo "• TensorCore library built and installed"
    echo "• Python bindings built and installed"
    echo "• Examples created in examples/ directory"
    echo
    echo "Next steps:"
    echo "• Run examples: python3 examples/basic_example.py"
    echo "• Run tests: python3 -m pytest tests/"
    echo "• Read documentation: docs/"
    echo
    echo "Happy learning with TensorCore! 🚀"
}

# Main build function
main() {
    echo "TensorCore Build Script"
    echo "======================"
    echo
    
    # Parse command line arguments
    SKIP_DEPS=false
    SKIP_TESTS=false
    SKIP_PYTHON=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-python)
                SKIP_PYTHON=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --skip-deps     Skip dependency installation"
                echo "  --skip-tests    Skip running tests"
                echo "  --skip-python   Skip Python bindings"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run build steps
    if [ "$SKIP_DEPS" = false ]; then
        check_dependencies
        install_python_deps
    fi
    
    create_build_dir
    configure_cmake
    build_library
    
    if [ "$SKIP_TESTS" = false ]; then
        run_tests
    fi
    
    install_library
    
    if [ "$SKIP_PYTHON" = false ]; then
        build_python_bindings
        
        if [ "$SKIP_TESTS" = false ]; then
            run_python_tests
        fi
    fi
    
    create_examples
    print_summary
}

# Run main function
main "$@"
