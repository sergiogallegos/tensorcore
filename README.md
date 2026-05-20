# TensorCore - Educational Machine Learning Library

A C++ machine learning library designed for educational purposes to understand the core mathematics and implementations behind popular ML libraries like NumPy, PyTorch, and TensorFlow.

## 🎯 Project Goals

- **Educational Focus**: Learn the fundamental mathematics and algorithms behind modern ML libraries
- **Performance**: Start with readable reference implementations, then add optional SIMD or BLAS-backed versions once correctness is locked down
- **Python Integration**: Provide seamless Python bindings for easy experimentation
- **Transparency**: Well-documented code showing exactly what happens under the hood
- **Modularity**: Clean, modular design that's easy to understand and extend

## Current Status

TensorCore is currently an educational prototype. The most useful path forward is
to keep a small, correct reference core and move broader framework-style features
behind an experimental boundary until they are implemented and tested.

### Working Reference Core

- Core tensor creation, storage, indexing, reshape, 2D transpose, scalar operations, and same-shape element-wise arithmetic
- Basic reductions without axes: `sum`, `mean`, `min`, `max`, `norm`
- Basic matrix operations: 2D `matmul`, 1D `dot`, and simple matrix inversion
- Common element-wise math functions, activation functions, and loss functions
- Early SIMD helpers for selected operations
- Standalone C++ assert-based tests for the current core

### Experimental Or Incomplete

- Axis reductions, arbitrary-axis transpose, broadcasting, and slicing
- Autograd integration with normal tensor operations
- Optimizers that update real parameter gradients
- Full neural-network training loops
- Pooling backward passes, LSTM, embeddings, BatchNorm/LayerNorm backward passes
- Scikit-learn style algorithms that depend on incomplete slicing/reduction behavior
- Python bindings for the full declared API
- BLAS, CUDA, distributed training, serialization, SVD/eigendecomposition, and advanced ML algorithms

Incomplete operations should throw clear `not implemented` errors instead of
returning placeholder values. See [TODO.md](TODO.md) for the active consolidation
plan.

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

- **Readable Tensor Core**: Small tensor type with explicit shape and storage behavior
- **Reference Math Implementations**: Straightforward loops for core math before advanced optimization
- **Selected SIMD Helpers**: Early vectorized paths for a few operations
- **Educational API Surface**: Work in progress toward a compact, teachable ML library
- **Performance Benchmarking**: Basic standalone benchmark executable

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
│   ├── sklearn.hpp             # Scikit-learn style ML algorithms
│   ├── simd_utils.hpp          # SIMD optimizations
│   ├── memory_pool.hpp         # Memory management
│   └── utils.hpp               # Utility functions
├── src/                        # C++ implementation
│   ├── tensor.cpp
│   ├── operations.cpp
│   ├── activations.cpp
│   ├── losses.cpp
│   ├── optimizers.cpp
│   ├── layers.cpp
│   ├── autograd.cpp
│   ├── sklearn.cpp             # Scikit-learn style ML algorithms
│   ├── simd_utils.cpp          # SIMD optimizations
│   └── memory_pool.cpp         # Memory management
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

The default CMake build targets the stable reference core. Broader modules such
as autograd, neural-network layers, optimizers, sklearn-style APIs, memory-pool
experiments, and Python bindings are currently experimental:

```bash
cmake .. -DTENSORCORE_BUILD_EXPERIMENTAL=ON
```

### Quick Start Example

#### Core Tensor Example
```cpp
#include "tensorcore/tensor.hpp"

using namespace tensorcore;

int main() {
    Tensor a = {{1.0, 2.0}, {3.0, 4.0}};
    Tensor b = {{5.0, 6.0}, {7.0, 8.0}};

    Tensor c = a.matmul(b);
    Tensor total = c.sum();

    c.print();
    total.print();

    return 0;
}
```

#### Scikit-learn Style Machine Learning

This API is experimental while slicing, broadcasting, and axis reductions are
being completed. Treat the example below as a design sketch, not a stable path.

```cpp
#include "tensorcore/sklearn.hpp"

using namespace tensorcore::sklearn;

int main() {
    // Create sample data
    Tensor X = random_normal({100, 4}, 0.0, 1.0);
    Tensor y = random_normal({100}, 0.0, 1.0);
    
    // Split data
    auto [X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.2);
    
    // Preprocess data
    StandardScaler scaler;
    scaler.fit(X_train);
    X_train = scaler.transform(X_train);
    X_test = scaler.transform(X_test);
    
    // Train linear regression
    LinearRegression lr;
    lr.fit(X_train, y_train);
    Tensor predictions = lr.predict(X_test);
    
    // Evaluate model
    double mse = mean_squared_error(y_test, predictions);
    double r2 = r2_score(y_test, predictions);
    
    std::cout << "MSE: " << mse << ", R²: " << r2 << std::endl;
    
    // Train decision tree
    DecisionTreeClassifier dt;
    dt.fit(X_train, y_train);
    Tensor tree_predictions = dt.predict(X_test);
    
    double accuracy = accuracy_score(y_test, tree_predictions);
    std::cout << "Decision Tree Accuracy: " << accuracy << std::endl;
    
    return 0;
}
```

## Core Features

### Stable Enough To Study

- Tensor creation, shape metadata, indexing, reshape, 2D transpose, and copy/move behavior
- Same-shape arithmetic and scalar arithmetic
- Basic element-wise math, activations, losses, and simple linear algebra
- Standalone tests and benchmarks for the current reference core

### Under Construction

- Autograd and optimizer design
- Neural-network layers beyond simple forward-pass experiments
- Scikit-learn style algorithms
- Python bindings for the complete C++ surface
- Advanced decomposition, GPU, and distributed features

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

## 🎓 Educational Machine Learning Features

### **Traditional Machine Learning**
- **Linear Models**: Understanding the mathematics behind linear regression, ridge, lasso
- **Tree-based Learning**: Decision trees, random forests, gradient boosting
- **Support Vector Machines**: Kernel methods and margin maximization
- **Naive Bayes**: Probabilistic classification with independence assumptions
- **Clustering**: K-means, DBSCAN, hierarchical clustering algorithms
- **Ensemble Methods**: Voting, bagging, boosting, stacking techniques

### **Deep Learning Foundations**
- **Recurrent Networks**: LSTM, GRU for sequential data processing
- **Attention Mechanisms**: Self-attention, multi-head attention, transformer architecture
- **Convolutional Networks**: Advanced CNN architectures, residual connections
- **Generative Models**: Variational Autoencoders, Generative Adversarial Networks
- **Reinforcement Learning**: Q-learning, policy gradient methods

### **Advanced Mathematics**
- **Matrix Decompositions**: SVD, QR, Cholesky, eigenvalue decomposition
- **Dimensionality Reduction**: PCA, LDA, t-SNE, UMAP
- **Signal Processing**: Fourier transforms, wavelet analysis, spectral methods
- **Optimization Theory**: Convex optimization, constrained optimization, duality
- **Information Theory**: Entropy, mutual information, KL divergence

## 🔧 Development

### Running Tests

```bash
# Build and run core functionality tests
mkdir build && cd build
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/tensor.cpp -o tensor.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/operations.cpp -o operations.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/autograd.cpp -o autograd.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/layers.cpp -o layers.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/optimizers.cpp -o optimizers.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/activations.cpp -o activations.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/simd_utils.cpp -o simd_utils.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../src/memory_pool.cpp -o memory_pool.o
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../test_core_functionality.cpp -o test_core.o
g++ -std=c++17 -mavx2 -mfma -o test_core test_core.o tensor.o operations.o autograd.o layers.o optimizers.o activations.o simd_utils.o memory_pool.o
./test_core

# Run SIMD performance tests
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../test_simd_performance.cpp -o test_simd.o
g++ -std=c++17 -mavx2 -mfma -o test_simd test_simd.o tensor.o operations.o autograd.o layers.o optimizers.o activations.o simd_utils.o memory_pool.o
./test_simd

# Run Conv2D tests
g++ -std=c++17 -mavx2 -mfma -I ../include -c ../test_conv2d.cpp -o test_conv2d.o
g++ -std=c++17 -mavx2 -mfma -o test_conv2d test_conv2d.o tensor.o operations.o autograd.o layers.o optimizers.o activations.o simd_utils.o memory_pool.o
./test_conv2d

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
# Run SIMD performance benchmarks
./test_simd

# Run Conv2D performance tests
./test_conv2d

# Run core functionality tests
./test_core

# Compare with NumPy (when Python bindings are ready)
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
- [SIMD Optimizations Explained](docs/internals/simd_optimizations.md)
- [Memory Management in C++](docs/internals/memory_management.md)

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
- Inspired by BLAS-style linear algebra interfaces; external BLAS integration is still experimental
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
