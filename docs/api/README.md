# TensorCore API Documentation

Welcome to the comprehensive API documentation for TensorCore, an educational machine learning library designed to help you understand the mathematical foundations behind popular libraries like NumPy, PyTorch, and TensorFlow.

## 📚 Documentation Overview

This documentation provides detailed explanations for every function, class, and operation in TensorCore, with a focus on educational value and mathematical understanding.

## 🗂️ API Reference

### Core Operations

1. **[Tensor Creation Functions](tensor_creation.md)**
   - `tensor()`, `zeros()`, `ones()`, `eye()`
   - `random_normal()`, `random_uniform()`
   - `arange()`, `linspace()`
   - `create_tensor()`, `set_seed()`

2. **[Basic Arithmetic Operations](basic_operations.md)**
   - Element-wise: `add()`, `subtract()`, `multiply()`, `divide()`
   - Scalar: `add_scalar()`, `multiply_scalar()`, etc.
   - In-place: `operator+=`, `operator-=`, etc.
   - Comparison: `equal()`, `less()`, `greater()`, etc.

3. **[Mathematical Functions](mathematical_functions.md)**
   - Trigonometric: `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`
   - Hyperbolic: `sinh()`, `cosh()`, `tanh()`, `asinh()`, `acosh()`, `atanh()`
   - Exponential: `exp()`, `log()`, `log2()`, `log10()`, `exp2()`, `expm1()`
   - Power: `sqrt()`, `cbrt()`, `square()`, `power()`
   - Rounding: `floor()`, `ceil()`, `round()`, `trunc()`
   - Absolute: `abs()`, `fabs()`, `sign()`, `copysign()`

4. **[Linear Algebra Operations](linear_algebra.md)**
   - Matrix: `matmul()`, `transpose()`, `inv()`, `pinv()`
   - Vector: `dot()`, `outer()`, `cross()`, `norm()`
   - Decompositions: `eig()`, `eigh()`, `svd()`
   - Properties: `trace()`, `det()`
   - Solving: `solve()`, `lstsq()`

5. **[Reduction Operations](reduction_operations.md)**
   - Basic: `sum()`, `mean()`, `max()`, `min()`, `prod()`
   - Statistical: `std()`, `var()`
   - Index-based: `argmax()`, `argmin()`
   - Multi-axis: Support for multiple axes

6. **[Shape Operations](shape_operations.md)**
   - Reshaping: `reshape()`, `transpose()`, `squeeze()`, `unsqueeze()`
   - Broadcasting: `broadcast_to()`, `is_broadcastable()`
   - Indexing: `slice()`, `index()`
   - Concatenation: `concatenate()`, `stack()`, `split()`
   - Padding: `pad()`, `tile()`, `repeat()`

### Machine Learning Components

7. **[Activation Functions](activation_functions.md)**
   - Basic: `relu()`, `leaky_relu()`, `elu()`, `gelu()`, `swish()`
   - Advanced: `sigmoid()`, `tanh()`, `softmax()`, `log_softmax()`
   - Specialized: `mish()`, `hard_sigmoid()`, `hard_tanh()`, `selu()`

8. **[Loss Functions](loss_functions.md)**
   - Regression: `mse_loss()`, `mae_loss()`, `huber_loss()`, `smooth_l1_loss()`
   - Classification: `cross_entropy_loss()`, `binary_cross_entropy_loss()`
   - Specialized: `hinge_loss()`, `kl_divergence_loss()`, `js_divergence_loss()`

9. **[Optimizer Classes](optimizer_classes.md)**
   - Basic: `SGD`, `Adam`, `RMSprop`, `Adagrad`
   - Advanced: `AdamW`, `AdaDelta`, `AdaMax`
   - Adaptive: `RAdam`, `Lion`

10. **[Neural Network Layers](neural_network_layers.md)**
    - Linear: `Dense`, `Linear`
    - Convolutional: `Conv2D`, `Conv1d`, `ConvTranspose2d`
    - Pooling: `MaxPool2D`, `AvgPool2D`
    - Recurrent: `LSTM`, `GRU`
    - Normalization: `BatchNorm`, `LayerNorm`
    - Regularization: `Dropout`, `Dropout2d`
    - Sequential: `Sequential` container for multi-layer networks

### Performance and Optimization

11. **[SIMD Optimizations](../internals/simd_optimizations.md)**
    - AVX2/AVX/SSE vectorized operations
    - CPU feature detection
    - Performance benchmarking
    - Memory alignment

12. **[Memory Management](../internals/memory_management.md)**
    - Memory pool system
    - Efficient allocation/deallocation
    - RAII wrappers
    - Performance monitoring

## 🎯 Educational Focus

Each function in this documentation includes:

- **Mathematical Definition**: The precise mathematical formula
- **Purpose**: Why this function exists and when to use it
- **Parameters**: Detailed explanation of each parameter
- **Examples**: Practical code examples showing usage
- **Performance Notes**: Computational complexity and memory usage
- **Best Practices**: How to use the function effectively
- **Common Patterns**: Real-world usage examples

## 🚀 Getting Started

### Quick Example

```cpp
#include <tensorcore/tensor.hpp>
#include <tensorcore/operations.hpp>
#include <tensorcore/nn/layers.hpp>

int main() {
    // Create tensors
    Tensor x = {1, 2, 3, 4, 5};
    Tensor y = {2, 3, 4, 5, 6};
    
    // Basic operations
    Tensor z = x + y;  // {3, 5, 7, 9, 11}
    Tensor w = x * 2;  // {2, 4, 6, 8, 10}
    
    // Mathematical functions
    Tensor a = exp(x);  // {2.718, 7.389, 20.086, 54.598, 148.413}
    Tensor b = sin(x);  // {0.841, 0.909, 0.141, -0.757, -0.959}
    
    // Linear algebra
    Tensor c = dot(x, y);  // 70
    Tensor d = norm(x);    // 7.416
    
    // Neural network
    Linear layer(5, 3);
    Tensor output = layer.forward(x);  // (3,)
    
    return 0;
}
```

### Learning Path

1. **Start with Tensor Creation**: Learn how to create and initialize tensors
2. **Master Basic Operations**: Understand element-wise and scalar operations
3. **Explore Mathematical Functions**: Learn about mathematical transformations
4. **Study Linear Algebra**: Understand matrix operations and decompositions
5. **Practice Reductions**: Learn about aggregation operations
6. **Work with Shapes**: Master tensor manipulation and broadcasting
7. **Build Neural Networks**: Combine everything to create ML models

## 📖 Reading the Documentation

### Function Documentation Format

Each function follows this structure:

```cpp
/**
 * @brief Brief description of the function
 * 
 * @details Detailed explanation of what the function does,
 * including mathematical definitions and use cases.
 * 
 * @param param1 Description of first parameter
 * @param param2 Description of second parameter
 * @return Description of return value
 * 
 * @throws ExceptionType Description of when this exception is thrown
 * 
 * @example
 * ```cpp
 * // Example code showing how to use the function
 * Tensor result = function_name(input1, input2);
 * ```
 * 
 * @see related_function1, related_function2
 * @since 1.0.0
 */
```

### Understanding Examples

- **Basic Usage**: Shows the simplest way to use the function
- **Advanced Usage**: Demonstrates more complex scenarios
- **Real-world Patterns**: Shows how the function fits into larger workflows
- **Performance Tips**: Examples of efficient usage

## 🔧 Implementation Details

### Memory Layout

TensorCore uses **row-major (C-style)** memory layout for compatibility with BLAS operations:

```cpp
// For a 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
// Memory layout: [1, 2, 3, 4, 5, 6]
// Index mapping: [i, j] -> i * cols + j
```

### Broadcasting Rules

TensorCore follows NumPy-style broadcasting:

1. **Dimension Alignment**: Dimensions are aligned from the right
2. **Size 1 Broadcasting**: Dimensions of size 1 can broadcast to any size
3. **Missing Dimensions**: Missing dimensions are treated as size 1

### Performance Characteristics

- **SIMD Vectorization**: AVX2/AVX/SSE instructions for 4x-8x performance boost
- **Memory Pool**: Efficient allocation/deallocation system for large tensors
- **CPU Feature Detection**: Automatic detection of available SIMD instructions
- **Memory Efficiency**: In-place operations and lazy evaluation where possible
- **Numerical Stability**: Careful implementation to prevent overflow/underflow
- **Performance Benchmarking**: Comprehensive testing framework for optimization

## 🎓 Learning Resources

### Mathematical Background

- **Linear Algebra**: Essential for understanding matrix operations
- **Calculus**: Important for gradients and optimization
- **Probability**: Useful for understanding loss functions and regularization
- **Numerical Analysis**: Helps understand numerical stability and precision

### Related Libraries

- **NumPy**: Python's fundamental package for scientific computing
- **PyTorch**: Deep learning framework with dynamic computation graphs
- **TensorFlow**: Machine learning platform with static computation graphs
- **Eigen**: C++ template library for linear algebra

## 🤝 Contributing

This documentation is part of the TensorCore educational project. If you find errors or have suggestions for improvement, please contribute to the project.

## 📄 License

This documentation is part of the TensorCore project and follows the same license terms.

---

**Happy Learning!** 🚀

Use this documentation to understand not just *how* to use TensorCore, but *why* each function works the way it does. The goal is to build deep understanding of the mathematical foundations that power modern machine learning libraries.
