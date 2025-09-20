# Tensor Creation Functions

This document provides comprehensive documentation for all tensor creation functions in TensorCore.

## Table of Contents

1. [Basic Tensor Creation](#basic-tensor-creation)
2. [Special Tensor Creation](#special-tensor-creation)
3. [Random Tensor Creation](#random-tensor-creation)
4. [Range and Sequence Creation](#range-and-sequence-creation)
5. [Utility Functions](#utility-functions)

## Basic Tensor Creation

### `tensor`

```cpp
/**
 * @brief Creates a tensor from initializer list data
 * 
 * @details This is the primary constructor for creating tensors from data.
 * The function automatically infers the shape from the input data structure.
 * For 1D data, it creates a vector tensor. For 2D data, it creates a matrix.
 * 
 * @param data Initializer list containing the tensor data
 * @return Tensor with automatically inferred shape
 * 
 * @throws std::invalid_argument if data is empty
 * @throws ShapeError if data structure is inconsistent
 * 
 * @example
 * ```cpp
 * // 1D tensor (vector)
 * Tensor a = tensor({1, 2, 3, 4});  // Shape: (4,)
 * 
 * // 2D tensor (matrix)
 * Tensor b = tensor({{1, 2, 3}, {4, 5, 6}});  // Shape: (2, 3)
 * 
 * // 3D tensor
 * Tensor c = tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});  // Shape: (2, 2, 2)
 * ```
 * 
 * @see zeros, ones, eye
 * @since 1.0.0
 */
Tensor tensor(std::initializer_list<value_type> data);
Tensor tensor(std::initializer_list<std::initializer_list<value_type>> data);
```

### `zeros`

```cpp
/**
 * @brief Creates a tensor filled with zeros
 * 
 * @details Creates a tensor of the specified shape with all elements set to 0.0.
 * This is commonly used for initializing parameters, creating masks, or as a
 * starting point for accumulation operations.
 * 
 * @param shape The desired shape of the tensor
 * @return Tensor filled with zeros
 * 
 * @throws std::invalid_argument if shape is empty or contains zero dimensions
 * 
 * @example
 * ```cpp
 * // 1D zero vector
 * Tensor a = zeros({5});  // [0, 0, 0, 0, 0]
 * 
 * // 2D zero matrix
 * Tensor b = zeros({3, 4});  // 3x4 matrix of zeros
 * 
 * // 3D zero tensor
 * Tensor c = zeros({2, 3, 4});  // 2x3x4 tensor of zeros
 * ```
 * 
 * @see ones, eye, random_normal
 * @since 1.0.0
 */
Tensor zeros(const shape_type& shape);
```

### `ones`

```cpp
/**
 * @brief Creates a tensor filled with ones
 * 
 * @details Creates a tensor of the specified shape with all elements set to 1.0.
 * This is useful for initializing parameters, creating identity-like structures,
 * or as a multiplicative identity in operations.
 * 
 * @param shape The desired shape of the tensor
 * @return Tensor filled with ones
 * 
 * @throws std::invalid_argument if shape is empty or contains zero dimensions
 * 
 * @example
 * ```cpp
 * // 1D ones vector
 * Tensor a = ones({5});  // [1, 1, 1, 1, 1]
 * 
 * // 2D ones matrix
 * Tensor b = ones({3, 4});  // 3x4 matrix of ones
 * 
 * // Broadcasting with ones
 * Tensor c = ones({1, 4});  // Can broadcast to (3, 4)
 * ```
 * 
 * @see zeros, eye, random_normal
 * @since 1.0.0
 */
Tensor ones(const shape_type& shape);
```

### `eye`

```cpp
/**
 * @brief Creates an identity matrix
 * 
 * @details Creates a square identity matrix where the diagonal elements are 1.0
 * and all off-diagonal elements are 0.0. The identity matrix has the property
 * that A @ I = I @ A = A for any compatible matrix A.
 * 
 * @param size The size of the square matrix (size x size)
 * @return Square identity matrix
 * 
 * @throws std::invalid_argument if size is zero
 * 
 * @example
 * ```cpp
 * // 3x3 identity matrix
 * Tensor I = eye(3);
 * // [[1, 0, 0],
 * //  [0, 1, 0],
 * //  [0, 0, 1]]
 * 
 * // 4x4 identity matrix
 * Tensor I4 = eye(4);
 * ```
 * 
 * @see zeros, ones, random_normal
 * @since 1.0.0
 */
Tensor eye(size_type size);

/**
 * @brief Creates a rectangular identity-like matrix
 * 
 * @details Creates a matrix where the main diagonal elements are 1.0 and all
 * other elements are 0.0. If the matrix is not square, only the diagonal
 * elements up to min(rows, cols) are set to 1.0.
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Rectangular matrix with ones on the diagonal
 * 
 * @throws std::invalid_argument if rows or cols is zero
 * 
 * @example
 * ```cpp
 * // 3x4 rectangular identity
 * Tensor I = eye(3, 4);
 * // [[1, 0, 0, 0],
 * //  [0, 1, 0, 0],
 * //  [0, 0, 1, 0]]
 * ```
 * 
 * @see eye(size_type), zeros, ones
 * @since 1.0.0
 */
Tensor eye(size_type rows, size_type cols);
```

## Random Tensor Creation

### `random_normal`

```cpp
/**
 * @brief Creates a tensor with normally distributed random values
 * 
 * @details Generates a tensor filled with random values drawn from a normal
 * (Gaussian) distribution with specified mean and standard deviation. This is
 * commonly used for weight initialization in neural networks.
 * 
 * @param shape The desired shape of the tensor
 * @param mean The mean of the normal distribution (default: 0.0)
 * @param std The standard deviation of the normal distribution (default: 1.0)
 * @return Tensor filled with random normal values
 * 
 * @throws std::invalid_argument if shape is empty or contains zero dimensions
 * @throws std::invalid_argument if std is negative
 * 
 * @example
 * ```cpp
 * // Standard normal distribution (mean=0, std=1)
 * Tensor a = random_normal({1000});  // 1000 random values
 * 
 * // Custom normal distribution
 * Tensor b = random_normal({3, 4}, 5.0, 2.0);  // mean=5, std=2
 * 
 * // Weight initialization for neural networks
 * Tensor weights = random_normal({784, 128}, 0.0, 0.01);  // Xavier initialization
 * ```
 * 
 * @note The random number generator is seeded for reproducibility. Use set_seed()
 *       to control the random state.
 * 
 * @see random_uniform, set_seed
 * @since 1.0.0
 */
Tensor random_normal(const shape_type& shape, double mean = 0.0, double std = 1.0);
```

### `random_uniform`

```cpp
/**
 * @brief Creates a tensor with uniformly distributed random values
 * 
 * @details Generates a tensor filled with random values drawn from a uniform
 * distribution over the specified range [min, max). This is useful for
 * initialization, data augmentation, and Monte Carlo methods.
 * 
 * @param shape The desired shape of the tensor
 * @param min The minimum value (inclusive) of the uniform distribution (default: 0.0)
 * @param max The maximum value (exclusive) of the uniform distribution (default: 1.0)
 * @return Tensor filled with random uniform values
 * 
 * @throws std::invalid_argument if shape is empty or contains zero dimensions
 * @throws std::invalid_argument if min >= max
 * 
 * @example
 * ```cpp
 * // Standard uniform distribution [0, 1)
 * Tensor a = random_uniform({1000});  // 1000 random values in [0, 1)
 * 
 * // Custom uniform distribution
 * Tensor b = random_uniform({3, 4}, -1.0, 1.0);  // values in [-1, 1)
 * 
 * // Integer-like uniform distribution
 * Tensor c = random_uniform({10}, 0, 10);  // values in [0, 10)
 * ```
 * 
 * @note The random number generator is seeded for reproducibility. Use set_seed()
 *       to control the random state.
 * 
 * @see random_normal, set_seed
 * @since 1.0.0
 */
Tensor random_uniform(const shape_type& shape, double min = 0.0, double max = 1.0);
```

## Range and Sequence Creation

### `arange`

```cpp
/**
 * @brief Creates a tensor with values in a range
 * 
 * @details Creates a 1D tensor with values from start (inclusive) to stop
 * (exclusive), incrementing by step. This is similar to Python's range()
 * function but returns a tensor instead of an iterator.
 * 
 * @param start The starting value (inclusive)
 * @param stop The ending value (exclusive)
 * @param step The step size (default: 1.0)
 * @return 1D tensor with values in the specified range
 * 
 * @throws std::invalid_argument if step is zero
 * @throws std::invalid_argument if start and stop are incompatible with step
 * 
 * @example
 * ```cpp
 * // Basic range
 * Tensor a = arange(0, 5);  // [0, 1, 2, 3, 4]
 * 
 * // Range with step
 * Tensor b = arange(0, 10, 2);  // [0, 2, 4, 6, 8]
 * 
 * // Negative step
 * Tensor c = arange(5, 0, -1);  // [5, 4, 3, 2, 1]
 * 
 * // Float range
 * Tensor d = arange(0.0, 1.0, 0.2);  // [0.0, 0.2, 0.4, 0.6, 0.8]
 * ```
 * 
 * @see linspace, zeros, ones
 * @since 1.0.0
 */
Tensor arange(double start, double stop, double step = 1.0);
```

### `linspace`

```cpp
/**
 * @brief Creates a tensor with linearly spaced values
 * 
 * @details Creates a 1D tensor with num evenly spaced values from start to stop
 * (both inclusive). This is useful for creating coordinate grids, plotting
 * ranges, and numerical analysis.
 * 
 * @param start The starting value (inclusive)
 * @param stop The ending value (inclusive)
 * @param num The number of values to generate (default: 50)
 * @return 1D tensor with linearly spaced values
 * 
 * @throws std::invalid_argument if num is less than 2
 * 
 * @example
 * ```cpp
 * // Basic linear spacing
 * Tensor a = linspace(0, 1, 5);  // [0.0, 0.25, 0.5, 0.75, 1.0]
 * 
 * // Coordinate grid
 * Tensor x = linspace(-2, 2, 100);  // 100 points from -2 to 2
 * 
 * // Logarithmic-like spacing (for plotting)
 * Tensor b = linspace(0.1, 10, 20);  // 20 points from 0.1 to 10
 * ```
 * 
 * @note Unlike arange(), linspace() includes both start and stop values.
 * 
 * @see arange, zeros, ones
 * @since 1.0.0
 */
Tensor linspace(double start, double stop, size_type num = 50);
```

## Utility Functions

### `create_tensor`

```cpp
/**
 * @brief Convenience function for creating tensors with optional parameters
 * 
 * @details This is a Python-friendly wrapper that provides a more flexible
 * interface for tensor creation. It automatically handles shape inference
 * and data type conversion.
 * 
 * @param data The data to create the tensor from
 * @param shape Optional shape specification (if None, inferred from data)
 * @param dtype Optional data type (currently only double supported)
 * @param requires_grad Whether the tensor requires gradients (default: false)
 * @return Tensor created from the input data
 * 
 * @throws std::invalid_argument if data is empty
 * @throws ShapeError if shape is incompatible with data
 * 
 * @example
 * ```cpp
 * // From list with inferred shape
 * Tensor a = create_tensor({1, 2, 3, 4});  // Shape: (4,)
 * 
 * // From list with explicit shape
 * Tensor b = create_tensor({1, 2, 3, 4}, {2, 2});  // Shape: (2, 2)
 * 
 * // With gradient tracking
 * Tensor c = create_tensor({1, 2, 3}, requires_grad=true);
 * ```
 * 
 * @see tensor, zeros, ones
 * @since 1.0.0
 */
Tensor create_tensor(const std::vector<double>& data, 
                    const std::vector<size_t>& shape = {}, 
                    bool requires_grad = false);
```

### `set_seed`

```cpp
/**
 * @brief Sets the random seed for reproducible results
 * 
 * @details Sets the seed for the random number generator used by random_normal()
 * and random_uniform(). This ensures that subsequent random operations produce
 * the same sequence of values, which is crucial for reproducible experiments.
 * 
 * @param seed The seed value
 * 
 * @example
 * ```cpp
 * // Set seed for reproducibility
 * set_seed(42);
 * 
 * // These will always produce the same values
 * Tensor a = random_normal({5});
 * Tensor b = random_uniform({5});
 * 
 * // Reset seed
 * set_seed(123);
 * // New random values will be generated
 * ```
 * 
 * @see random_normal, random_uniform
 * @since 1.0.0
 */
void set_seed(unsigned int seed);
```

## Mathematical Properties

### Memory Layout
All tensors use **row-major (C-style)** memory layout for compatibility with BLAS operations:

```cpp
// For a 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
// Memory layout: [1, 2, 3, 4, 5, 6]
// Index mapping: [i, j] -> i * cols + j
```

### Broadcasting Rules
When creating tensors for operations, keep in mind the broadcasting rules:

1. **Dimension Alignment**: Dimensions are aligned from the right
2. **Size 1 Broadcasting**: Dimensions of size 1 can broadcast to any size
3. **Missing Dimensions**: Missing dimensions are treated as size 1

```cpp
// Examples of broadcasting-compatible shapes
Tensor a = zeros({3, 1, 4});  // Can broadcast to (3, 2, 4)
Tensor b = zeros({2, 4});     // Can broadcast to (3, 2, 4)
Tensor c = a + b;             // Result shape: (3, 2, 4)
```

### Performance Considerations

- **Memory Allocation**: Large tensors are allocated in contiguous memory blocks
- **Copy Semantics**: Tensor copying is done lazily when possible
- **Random Generation**: Uses efficient Mersenne Twister algorithm
- **Shape Validation**: Minimal overhead for shape checking

## Common Patterns

### Weight Initialization
```cpp
// Xavier/Glorot initialization for neural networks
Tensor weights = random_normal({input_size, output_size}, 0.0, 
                              std::sqrt(2.0 / (input_size + output_size)));

// He initialization for ReLU networks
Tensor weights = random_normal({input_size, output_size}, 0.0, 
                              std::sqrt(2.0 / input_size));
```

### Coordinate Grids
```cpp
// 2D coordinate grid
Tensor x = linspace(-1, 1, 100);
Tensor y = linspace(-1, 1, 100);
// Use for plotting or function evaluation
```

### Mask Creation
```cpp
// Create a mask for valid elements
Tensor mask = zeros({batch_size, sequence_length});
// Set valid positions to 1
for (int i = 0; i < valid_length; ++i) {
    mask[0, i] = 1.0;
}
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's tensor creation functions for their machine learning projects.
