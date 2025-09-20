# TensorCore - Educational Machine Learning Library

A C++ machine learning library designed for educational purposes to understand the core mathematics and implementations behind popular ML libraries like NumPy, PyTorch, and TensorFlow.

## 🎯 Project Goals

- **Educational Focus**: Learn the fundamental mathematics and algorithms behind modern ML libraries
- **Performance**: Implement efficient C++ code with SIMD optimizations and BLAS integration
- **Python Integration**: Provide seamless Python bindings for easy experimentation
- **Transparency**: Well-documented code showing exactly what happens under the hood
- **Modularity**: Clean, modular design that's easy to understand and extend

## ✅ Current Status

### **What's Working Now**
- ✅ **Core Tensor Operations**: Creation, manipulation, arithmetic, shape operations
- ✅ **Mathematical Functions**: 50+ mathematical operations (sin, cos, exp, log, etc.)
- ✅ **Activation Functions**: 15+ activation functions (ReLU, Sigmoid, Tanh, Softmax, etc.)
- ✅ **Loss Functions**: 10+ loss functions (MSE, MAE, Cross-Entropy, etc.)
- ✅ **Utility Functions**: Tensor creation, random generation, memory management
- ✅ **Comprehensive Testing**: Unit tests, performance benchmarks, edge case testing
- ✅ **Documentation**: Complete API documentation with mathematical explanations

### **What's Coming Next**
- 🚧 **Automatic Differentiation**: Forward and reverse mode AD for backpropagation
- 🚧 **Neural Network Layers**: Dense, Convolutional, Recurrent layers
- 🚧 **Optimizers**: SGD, Adam, RMSprop, and other optimization algorithms
- 🚧 **Python Bindings**: Full Python integration for easy experimentation
- 🚧 **GPU Support**: CUDA integration for accelerated computing

*See [Development Roadmap](README_ROADMAP.md) for complete feature list and timeline.*

## 🏗️ Architecture Overview

### Core Components

- **Tensor Operations**: Multi-dimensional array operations with broadcasting
- **Mathematical Functions**: Linear algebra, statistics, and numerical operations
- **Activation Functions**: ReLU, Sigmoid, Tanh, and other activation functions
- **Loss Functions**: MSE, Cross-Entropy, and other common loss functions
- **Optimizers**: SGD, Adam, RMSprop, and other optimization algorithms
- **Neural Network Layers**: Dense, Convolutional, and other layer types

### Key Features

- **SIMD Optimizations**: Vectorized operations for maximum performance
- **BLAS Integration**: Leverage optimized linear algebra libraries
- **Memory Management**: Efficient memory allocation and deallocation
- **Gradient Computation**: Automatic differentiation for backpropagation
- **GPU Support**: CUDA integration for GPU acceleration (future)

## 📁 Project Structure

```
tensorcore/
├── include/tensorcore/          # Public C++ headers
│   ├── tensor.hpp              # Core tensor class
│   ├── operations.hpp          # Mathematical operations
│   ├── activations.hpp         # Activation functions
│   ├── losses.hpp              # Loss functions
│   ├── optimizers.hpp          # Optimization algorithms
│   ├── layers.hpp              # Neural network layers
│   ├── autograd.hpp            # Automatic differentiation
│   └── utils.hpp               # Utility functions
├── src/                        # C++ implementation
│   ├── tensor.cpp
│   ├── operations.cpp
│   ├── activations.cpp
│   ├── losses.cpp
│   ├── optimizers.cpp
│   ├── layers.cpp
│   ├── autograd.cpp
│   ├── blas_wrapper.cpp        # BLAS integration
│   ├── simd_utils.cpp          # SIMD optimizations
│   └── memory_manager.cpp      # Memory management
├── python/                     # Python bindings
│   ├── __init__.py
│   ├── tensorcore_core.cpp     # pybind11 bindings
│   ├── setup.py               # Build configuration
│   └── requirements.txt
├── tests/                      # Unit tests
│   ├── test_tensor.cpp
│   ├── test_operations.cpp
│   ├── test_activations.cpp
│   └── test_python.py
├── benchmarks/                 # Performance tests
│   ├── tensor_benchmarks.cpp
│   └── operations_benchmarks.cpp
├── examples/                   # Usage examples
│   ├── basic_tensor_ops.py
│   ├── neural_network.py
│   └── linear_regression.py
├── docs/                       # Documentation
│   ├── api/                    # API documentation
│   ├── tutorials/              # Tutorial notebooks
│   └── internals/              # Internal implementation docs
├── scripts/                    # Build and utility scripts
│   ├── build.sh
│   ├── test.sh
│   └── benchmark.sh
├── CMakeLists.txt              # Main CMake configuration
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- C++17 or later
- CMake 3.15+
- Python 3.8+
- BLAS library (OpenBLAS, Intel MKL, or ATLAS)
- pybind11 (for Python bindings)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/tensorcore.git
cd tensorcore

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the library
make -j$(nproc)

# Run tests
make test

# Install Python bindings
cd ../python
pip install -e .
```

### Quick Start Example

```python
import tensorcore as tc

# Create tensors
a = tc.tensor([1, 2, 3, 4], shape=(2, 2))
b = tc.tensor([5, 6, 7, 8], shape=(2, 2))

# Perform operations
c = a + b
d = tc.matmul(a, b.T)

print("Addition:", c)
print("Matrix multiplication:", d)

# Neural network example
model = tc.nn.Sequential([
    tc.nn.Linear(784, 128),
    tc.nn.ReLU(),
    tc.nn.Linear(128, 10)
])

# Forward pass
x = tc.tensor(input_data)
output = model(x)
```

## 🧮 Mathematical Foundations

This library implements the core mathematical concepts behind machine learning:

### Linear Algebra
- Matrix operations (multiplication, addition, transposition)
- Vector operations (dot product, cross product)
- Eigenvalue decomposition
- Singular Value Decomposition (SVD)

### Calculus
- Automatic differentiation
- Gradient computation
- Chain rule implementation
- Backpropagation algorithms

### Statistics
- Probability distributions
- Statistical moments
- Sampling methods
- Hypothesis testing utilities

## 🔧 Development

### Running Tests

```bash
# C++ tests
cd build && make test

# Python tests
python -m pytest tests/

# All tests
./scripts/test.sh
```

### Benchmarking

```bash
# Run performance benchmarks
./scripts/benchmark.sh

# Compare with NumPy
python benchmarks/compare_with_numpy.py
```

### Code Style

- Follow Google C++ Style Guide
- Use clang-format for code formatting
- Write comprehensive unit tests
- Document all public APIs

## 📚 Learning Resources

### **🎓 Educational Guides**
- **[Educational Concepts & Theory](README_EDUCATIONAL.md)** - Deep dive into ML/DL concepts and why these libraries exist
- **[Development Roadmap](README_ROADMAP.md)** - Complete roadmap of pending features for a production-ready library
- **[Getting Started Tutorial](docs/tutorials/getting_started.md)** - Step-by-step introduction to TensorCore

### **📖 Technical Documentation**
- [Linear Algebra for Machine Learning](docs/tutorials/linear_algebra.md)
- [Understanding Automatic Differentiation](docs/tutorials/autograd.md)
- [SIMD Optimizations Explained](docs/tutorials/simd.md)
- [Memory Management in C++](docs/tutorials/memory.md)

### **🔬 API Documentation**
- [Tensor Creation Functions](docs/api/tensor_creation.md)
- [Mathematical Operations](docs/api/mathematical_functions.md)
- [Activation Functions](docs/api/activation_functions.md)
- [Loss Functions](docs/api/loss_functions.md)
- [Complete API Reference](docs/api/README.md)

## 🤝 Contributing

This is an educational project! Contributions are welcome, especially:

- Implementation of new mathematical operations
- Performance optimizations
- Educational examples and tutorials
- Documentation improvements
- Bug fixes and code quality improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by NumPy, PyTorch, and TensorFlow
- Built with pybind11 for Python integration
- Uses OpenBLAS for linear algebra operations
- Educational resources from various ML courses and textbooks

## 🎓 Educational Value

By building this library, you'll learn:

1. **Core Mathematics**: How linear algebra and calculus power ML algorithms
2. **Memory Management**: Efficient data structures and memory allocation
3. **Performance Optimization**: SIMD, vectorization, and parallel computing
4. **API Design**: How to create intuitive and efficient interfaces
5. **Python Integration**: Bridging high-level Python with low-level C++
6. **Testing and Benchmarking**: Ensuring correctness and performance

---

**Happy Learning! 🚀**

*This library is designed to be your gateway into understanding the beautiful mathematics and engineering behind modern machine learning frameworks.*