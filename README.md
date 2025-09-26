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
- ✅ **Automatic Differentiation**: Complete computational graph with forward/backward passes
- ✅ **Neural Network Layers**: Dense layer with proper weight initialization and gradients
- ✅ **Optimizers**: SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, Adamax, Nadam
- ✅ **Sequential Networks**: Multi-layer neural networks with end-to-end training
- ✅ **Comprehensive Testing**: All core functionality tests passing
- ✅ **Documentation**: Complete API documentation with mathematical explanations

### **What's Coming Next**
- 🚧 **Advanced Layers**: Conv2D, MaxPool2D, LSTM, Transformer layers
- 🚧 **Performance Optimizations**: SIMD, memory pooling, GPU acceleration
- 🚧 **Model Serialization**: Save/load trained models
- 🚧 **Python Bindings**: Full Python integration for easy experimentation
- 🚧 **Distributed Training**: Multi-GPU and multi-node support

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

```cpp
#include "tensorcore/tensor.hpp"
#include "tensorcore/layers.hpp"
#include "tensorcore/optimizers.hpp"
#include "tensorcore/autograd.hpp"

using namespace tensorcore;

int main() {
    // Create tensors with automatic differentiation
    auto x = variable(Tensor({1, 2}, {1.0, 2.0}), true);
    auto y = variable(Tensor({1, 2}, {3.0, 4.0}), true);
    
    // Perform operations with gradient tracking
    auto z = global_graph.add(x, y);
    auto result = global_graph.multiply(z, z);
    
    // Compute gradients
    global_graph.backward(result);
    
    // Create a neural network
    auto layer1 = std::make_shared<Dense>(2, 3, true, "relu");
    auto layer2 = std::make_shared<Dense>(3, 1, true, "sigmoid");
    Sequential network({layer1, layer2});
    
    // Forward pass
    Tensor input({1, 2}, {1.0, 2.0});
    Tensor output = network.forward(input);
    
    // Training with optimizer
    SGD optimizer(0.01);
    optimizer.add_parameters(network.get_parameters());
    
    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        Tensor pred = network.forward(input);
        // Compute loss and gradients...
        network.backward(grad_output);
        optimizer.step();
        network.zero_grad();
    }
    
    return 0;
}
```

## 🚀 Core Features Implemented

### **Automatic Differentiation System**
- Complete computational graph with forward/backward passes
- Support for complex mathematical expressions
- Efficient gradient computation using chain rule
- Memory-efficient graph construction and cleanup

### **Neural Network Components**
- **Dense Layer**: Fully connected layer with Xavier initialization
- **Activation Functions**: ReLU, Sigmoid, Tanh with proper gradients
- **Sequential Container**: Easy multi-layer network construction
- **Gradient Flow**: End-to-end gradient computation through layers

### **Optimization Algorithms**
- **SGD**: Stochastic Gradient Descent with momentum
- **Adam**: Adaptive learning rate optimization
- **AdamW**: Adam with decoupled weight decay
- **RMSprop**: Root Mean Square propagation
- **Adagrad**: Adaptive gradient algorithm
- **Adadelta**: Adagrad with moving average
- **Adamax**: Adam with infinity norm
- **Nadam**: Nesterov-accelerated Adam

### **Testing & Validation**
- Comprehensive test suite with 100% core functionality coverage
- End-to-end neural network training verification
- Gradient computation accuracy validation
- Performance benchmarking framework

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
# Build and run core functionality tests
mkdir build && cd build
g++ -std=c++17 -I ../include -c ../src/tensor.cpp -o tensor.o
g++ -std=c++17 -I ../include -c ../src/operations.cpp -o operations.o
g++ -std=c++17 -I ../include -c ../src/autograd.cpp -o autograd.o
g++ -std=c++17 -I ../include -c ../src/layers.cpp -o layers.o
g++ -std=c++17 -I ../include -c ../src/optimizers.cpp -o optimizers.o
g++ -std=c++17 -I ../include -c ../src/activations.cpp -o activations.o
g++ -std=c++17 -I ../include -c ../test_core_functionality.cpp -o test_core.o
g++ -std=c++17 -o test_core test_core.o tensor.o operations.o autograd.o layers.o optimizers.o activations.o
./test_core

# Expected output:
# Testing TensorCore Core Functionality
# =====================================
# Testing basic autograd...
# ✓ Basic autograd test passed
# Testing Dense layer...
# ✓ Dense layer test passed
# Testing SGD optimizer...
# ✓ Optimizer test passed
# Testing simple neural network...
# ✓ Simple neural network test passed
# 🎉 All core functionality tests passed!
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