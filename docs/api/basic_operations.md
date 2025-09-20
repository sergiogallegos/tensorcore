# Basic Arithmetic Operations

This document provides comprehensive documentation for all basic arithmetic operations in TensorCore.

## Table of Contents

1. [Element-wise Operations](#element-wise-operations)
2. [Scalar Operations](#scalar-operations)
3. [In-place Operations](#in-place-operations)
4. [Comparison Operations](#comparison-operations)
5. [Broadcasting](#broadcasting)
6. [Performance Considerations](#performance-considerations)

## Element-wise Operations

### `add`

```cpp
/**
 * @brief Computes element-wise addition of two tensors
 * 
 * @details This function performs element-wise addition between two tensors,
 * following NumPy-style broadcasting rules. The operation is mathematically
 * defined as:
 * 
 *     result[i] = a[i] + b[i]
 * 
 * where i ranges over all valid indices after broadcasting.
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @return Tensor containing element-wise sum
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * @throws DimensionError if tensors have incompatible dimensions
 * 
 * @example
 * ```cpp
 * // Same shape tensors
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {5, 6, 7, 8};
 * Tensor result = add(a, b);  // {6, 8, 10, 12}
 * 
 * // Broadcasting
 * Tensor c = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor d = {10, 20, 30};            // (3,)
 * Tensor result2 = add(c, d);         // {{11, 22, 33}, {14, 25, 36}}
 * ```
 * 
 * @see subtract, multiply, divide
 * @since 1.0.0
 */
Tensor add(const Tensor& a, const Tensor& b);
```

### `subtract`

```cpp
/**
 * @brief Computes element-wise subtraction of two tensors
 * 
 * @details This function performs element-wise subtraction between two tensors,
 * following NumPy-style broadcasting rules. The operation is mathematically
 * defined as:
 * 
 *     result[i] = a[i] - b[i]
 * 
 * where i ranges over all valid indices after broadcasting.
 * 
 * @param a First input tensor (minuend)
 * @param b Second input tensor (subtrahend)
 * @return Tensor containing element-wise difference
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * @throws DimensionError if tensors have incompatible dimensions
 * 
 * @example
 * ```cpp
 * // Same shape tensors
 * Tensor a = {5, 6, 7, 8};
 * Tensor b = {1, 2, 3, 4};
 * Tensor result = subtract(a, b);  // {4, 4, 4, 4}
 * 
 * // Broadcasting
 * Tensor c = {{10, 20, 30}, {40, 50, 60}};  // (2, 3)
 * Tensor d = {1, 2, 3};                     // (3,)
 * Tensor result2 = subtract(c, d);          // {{9, 18, 27}, {39, 48, 57}}
 * ```
 * 
 * @see add, multiply, divide
 * @since 1.0.0
 */
Tensor subtract(const Tensor& a, const Tensor& b);
```

### `multiply`

```cpp
/**
 * @brief Computes element-wise multiplication of two tensors
 * 
 * @details This function performs element-wise multiplication (Hadamard product)
 * between two tensors, following NumPy-style broadcasting rules. The operation
 * is mathematically defined as:
 * 
 *     result[i] = a[i] * b[i]
 * 
 * where i ranges over all valid indices after broadcasting.
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @return Tensor containing element-wise product
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * @throws DimensionError if tensors have incompatible dimensions
 * 
 * @example
 * ```cpp
 * // Same shape tensors
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {5, 6, 7, 8};
 * Tensor result = multiply(a, b);  // {5, 12, 21, 32}
 * 
 * // Broadcasting
 * Tensor c = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor d = {2, 3, 4};               // (3,)
 * Tensor result2 = multiply(c, d);    // {{2, 6, 12}, {8, 15, 24}}
 * ```
 * 
 * @note This is element-wise multiplication, not matrix multiplication.
 *       Use matmul() for matrix multiplication.
 * 
 * @see add, subtract, divide, matmul
 * @since 1.0.0
 */
Tensor multiply(const Tensor& a, const Tensor& b);
```

### `divide`

```cpp
/**
 * @brief Computes element-wise division of two tensors
 * 
 * @details This function performs element-wise division between two tensors,
 * following NumPy-style broadcasting rules. The operation is mathematically
 * defined as:
 * 
 *     result[i] = a[i] / b[i]
 * 
 * where i ranges over all valid indices after broadcasting.
 * 
 * @param a First input tensor (dividend)
 * @param b Second input tensor (divisor)
 * @return Tensor containing element-wise quotient
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * @throws DimensionError if tensors have incompatible dimensions
 * @throws std::runtime_error if division by zero occurs
 * 
 * @example
 * ```cpp
 * // Same shape tensors
 * Tensor a = {10, 20, 30, 40};
 * Tensor b = {2, 4, 5, 8};
 * Tensor result = divide(a, b);  // {5, 5, 6, 5}
 * 
 * // Broadcasting
 * Tensor c = {{12, 18, 24}, {30, 36, 42}};  // (2, 3)
 * Tensor d = {2, 3, 4};                     // (3,)
 * Tensor result2 = divide(c, d);            // {{6, 6, 6}, {15, 12, 10.5}}
 * ```
 * 
 * @see add, subtract, multiply
 * @since 1.0.0
 */
Tensor divide(const Tensor& a, const Tensor& b);
```

### `power`

```cpp
/**
 * @brief Computes element-wise power of two tensors
 * 
 * @details This function performs element-wise power operation between two tensors,
 * following NumPy-style broadcasting rules. The operation is mathematically
 * defined as:
 * 
 *     result[i] = a[i]^b[i]
 * 
 * where i ranges over all valid indices after broadcasting.
 * 
 * @param a First input tensor (base)
 * @param b Second input tensor (exponent)
 * @return Tensor containing element-wise power
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * @throws DimensionError if tensors have incompatible dimensions
 * @throws std::runtime_error if invalid power operation (e.g., 0^0)
 * 
 * @example
 * ```cpp
 * // Same shape tensors
 * Tensor a = {2, 3, 4, 5};
 * Tensor b = {2, 2, 2, 2};
 * Tensor result = power(a, b);  // {4, 9, 16, 25}
 * 
 * // Broadcasting
 * Tensor c = {{2, 3, 4}, {5, 6, 7}};  // (2, 3)
 * Tensor d = {2, 3, 4};               // (3,)
 * Tensor result2 = power(c, d);       // {{4, 27, 256}, {25, 216, 2401}}
 * ```
 * 
 * @see add, subtract, multiply, divide
 * @since 1.0.0
 */
Tensor power(const Tensor& a, const Tensor& b);
```

### `mod`

```cpp
/**
 * @brief Computes element-wise modulo of two tensors
 * 
 * @details This function performs element-wise modulo operation between two tensors,
 * following NumPy-style broadcasting rules. The operation is mathematically
 * defined as:
 * 
 *     result[i] = a[i] % b[i]
 * 
 * where i ranges over all valid indices after broadcasting.
 * 
 * @param a First input tensor (dividend)
 * @param b Second input tensor (divisor)
 * @return Tensor containing element-wise remainder
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * @throws DimensionError if tensors have incompatible dimensions
 * @throws std::runtime_error if division by zero occurs
 * 
 * @example
 * ```cpp
 * // Same shape tensors
 * Tensor a = {10, 20, 30, 40};
 * Tensor b = {3, 4, 5, 6};
 * Tensor result = mod(a, b);  // {1, 0, 0, 4}
 * 
 * // Broadcasting
 * Tensor c = {{7, 14, 21}, {28, 35, 42}};  // (2, 3)
 * Tensor d = {3, 4, 5};                    // (3,)
 * Tensor result2 = mod(c, d);              // {{1, 2, 1}, {1, 3, 2}}
 * ```
 * 
 * @see add, subtract, multiply, divide
 * @since 1.0.0
 */
Tensor mod(const Tensor& a, const Tensor& b);
```

## Scalar Operations

### `add_scalar`

```cpp
/**
 * @brief Adds a scalar value to all elements of a tensor
 * 
 * @details This function adds a scalar value to every element of the input tensor.
 * The operation is mathematically defined as:
 * 
 *     result[i] = tensor[i] + scalar
 * 
 * for all valid indices i.
 * 
 * @param tensor Input tensor
 * @param scalar Scalar value to add
 * @return Tensor with scalar added to each element
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor result = add_scalar(a, 10);  // {11, 12, 13, 14}
 * 
 * Tensor b = {{1, 2}, {3, 4}};
 * Tensor result2 = add_scalar(b, 5);  // {{6, 7}, {8, 9}}
 * ```
 * 
 * @see subtract_scalar, multiply_scalar, divide_scalar
 * @since 1.0.0
 */
Tensor add_scalar(const Tensor& tensor, double scalar);
```

### `subtract_scalar`

```cpp
/**
 * @brief Subtracts a scalar value from all elements of a tensor
 * 
 * @details This function subtracts a scalar value from every element of the
 * input tensor. The operation is mathematically defined as:
 * 
 *     result[i] = tensor[i] - scalar
 * 
 * for all valid indices i.
 * 
 * @param tensor Input tensor
 * @param scalar Scalar value to subtract
 * @return Tensor with scalar subtracted from each element
 * 
 * @example
 * ```cpp
 * Tensor a = {10, 20, 30, 40};
 * Tensor result = subtract_scalar(a, 5);  // {5, 15, 25, 35}
 * 
 * Tensor b = {{10, 20}, {30, 40}};
 * Tensor result2 = subtract_scalar(b, 10);  // {{0, 10}, {20, 30}}
 * ```
 * 
 * @see add_scalar, multiply_scalar, divide_scalar
 * @since 1.0.0
 */
Tensor subtract_scalar(const Tensor& tensor, double scalar);
```

### `multiply_scalar`

```cpp
/**
 * @brief Multiplies all elements of a tensor by a scalar value
 * 
 * @details This function multiplies every element of the input tensor by a
 * scalar value. The operation is mathematically defined as:
 * 
 *     result[i] = tensor[i] * scalar
 * 
 * for all valid indices i.
 * 
 * @param tensor Input tensor
 * @param scalar Scalar value to multiply by
 * @return Tensor with each element multiplied by scalar
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor result = multiply_scalar(a, 2);  // {2, 4, 6, 8}
 * 
 * Tensor b = {{1, 2}, {3, 4}};
 * Tensor result2 = multiply_scalar(b, 0.5);  // {{0.5, 1}, {1.5, 2}}
 * ```
 * 
 * @see add_scalar, subtract_scalar, divide_scalar
 * @since 1.0.0
 */
Tensor multiply_scalar(const Tensor& tensor, double scalar);
```

### `divide_scalar`

```cpp
/**
 * @brief Divides all elements of a tensor by a scalar value
 * 
 * @details This function divides every element of the input tensor by a
 * scalar value. The operation is mathematically defined as:
 * 
 *     result[i] = tensor[i] / scalar
 * 
 * for all valid indices i.
 * 
 * @param tensor Input tensor
 * @param scalar Scalar value to divide by
 * @return Tensor with each element divided by scalar
 * 
 * @throws std::runtime_error if scalar is zero
 * 
 * @example
 * ```cpp
 * Tensor a = {10, 20, 30, 40};
 * Tensor result = divide_scalar(a, 2);  // {5, 10, 15, 20}
 * 
 * Tensor b = {{10, 20}, {30, 40}};
 * Tensor result2 = divide_scalar(b, 5);  // {{2, 4}, {6, 8}}
 * ```
 * 
 * @see add_scalar, subtract_scalar, multiply_scalar
 * @since 1.0.0
 */
Tensor divide_scalar(const Tensor& tensor, double scalar);
```

### `power_scalar`

```cpp
/**
 * @brief Raises all elements of a tensor to a scalar power
 * 
 * @details This function raises every element of the input tensor to a
 * scalar power. The operation is mathematically defined as:
 * 
 *     result[i] = tensor[i]^scalar
 * 
 * for all valid indices i.
 * 
 * @param tensor Input tensor
 * @param scalar Scalar power
 * @return Tensor with each element raised to the scalar power
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor result = power_scalar(a, 2);  // {1, 4, 9, 16}
 * 
 * Tensor b = {{2, 3}, {4, 5}};
 * Tensor result2 = power_scalar(b, 3);  // {{8, 27}, {64, 125}}
 * ```
 * 
 * @see add_scalar, subtract_scalar, multiply_scalar, divide_scalar
 * @since 1.0.0
 */
Tensor power_scalar(const Tensor& tensor, double scalar);
```

## In-place Operations

### `operator+=`

```cpp
/**
 * @brief In-place addition operator
 * 
 * @details Performs in-place addition, modifying the left operand directly.
 * This is more memory efficient than creating a new tensor for the result.
 * 
 * @param other Tensor to add
 * @return Reference to the modified tensor
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {5, 6, 7, 8};
 * a += b;  // a is now {6, 8, 10, 12}
 * ```
 * 
 * @see operator-=, operator*=, operator/=
 * @since 1.0.0
 */
Tensor& operator+=(const Tensor& other);
```

### `operator-=`

```cpp
/**
 * @brief In-place subtraction operator
 * 
 * @details Performs in-place subtraction, modifying the left operand directly.
 * This is more memory efficient than creating a new tensor for the result.
 * 
 * @param other Tensor to subtract
 * @return Reference to the modified tensor
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor a = {10, 20, 30, 40};
 * Tensor b = {1, 2, 3, 4};
 * a -= b;  // a is now {9, 18, 27, 36}
 * ```
 * 
 * @see operator+=, operator*=, operator/=
 * @since 1.0.0
 */
Tensor& operator-=(const Tensor& other);
```

### `operator*=`

```cpp
/**
 * @brief In-place multiplication operator
 * 
 * @details Performs in-place multiplication, modifying the left operand directly.
 * This is more memory efficient than creating a new tensor for the result.
 * 
 * @param other Tensor to multiply by
 * @return Reference to the modified tensor
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {2, 3, 4, 5};
 * a *= b;  // a is now {2, 6, 12, 20}
 * ```
 * 
 * @see operator+=, operator-=, operator/=
 * @since 1.0.0
 */
Tensor& operator*=(const Tensor& other);
```

### `operator/=`

```cpp
/**
 * @brief In-place division operator
 * 
 * @details Performs in-place division, modifying the left operand directly.
 * This is more memory efficient than creating a new tensor for the result.
 * 
 * @param other Tensor to divide by
 * @return Reference to the modified tensor
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * @throws std::runtime_error if division by zero occurs
 * 
 * @example
 * ```cpp
 * Tensor a = {10, 20, 30, 40};
 * Tensor b = {2, 4, 5, 8};
 * a /= b;  // a is now {5, 5, 6, 5}
 * ```
 * 
 * @see operator+=, operator-=, operator*=
 * @since 1.0.0
 */
Tensor& operator/=(const Tensor& other);
```

## Comparison Operations

### `equal`

```cpp
/**
 * @brief Element-wise equality comparison
 * 
 * @details Performs element-wise equality comparison between two tensors,
 * returning a boolean tensor with the same shape. The operation is defined as:
 * 
 *     result[i] = (a[i] == b[i])
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @return Boolean tensor with element-wise equality results
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {1, 3, 2, 4};
 * Tensor result = equal(a, b);  // {true, false, false, true}
 * ```
 * 
 * @see not_equal, less, greater
 * @since 1.0.0
 */
Tensor equal(const Tensor& a, const Tensor& b);
```

### `not_equal`

```cpp
/**
 * @brief Element-wise inequality comparison
 * 
 * @details Performs element-wise inequality comparison between two tensors,
 * returning a boolean tensor with the same shape. The operation is defined as:
 * 
 *     result[i] = (a[i] != b[i])
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @return Boolean tensor with element-wise inequality results
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {1, 3, 2, 4};
 * Tensor result = not_equal(a, b);  // {false, true, true, false}
 * ```
 * 
 * @see equal, less, greater
 * @since 1.0.0
 */
Tensor not_equal(const Tensor& a, const Tensor& b);
```

### `less`

```cpp
/**
 * @brief Element-wise less-than comparison
 * 
 * @details Performs element-wise less-than comparison between two tensors,
 * returning a boolean tensor with the same shape. The operation is defined as:
 * 
 *     result[i] = (a[i] < b[i])
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @return Boolean tensor with element-wise less-than results
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {2, 1, 4, 3};
 * Tensor result = less(a, b);  // {true, false, true, false}
 * ```
 * 
 * @see equal, not_equal, greater
 * @since 1.0.0
 */
Tensor less(const Tensor& a, const Tensor& b);
```

### `greater`

```cpp
/**
 * @brief Element-wise greater-than comparison
 * 
 * @details Performs element-wise greater-than comparison between two tensors,
 * returning a boolean tensor with the same shape. The operation is defined as:
 * 
 *     result[i] = (a[i] > b[i])
 * 
 * @param a First input tensor
 * @param b Second input tensor
 * @return Boolean tensor with element-wise greater-than results
 * 
 * @throws ShapeError if tensors cannot be broadcast together
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * Tensor b = {2, 1, 4, 3};
 * Tensor result = greater(a, b);  // {false, true, false, true}
 * ```
 * 
 * @see equal, not_equal, less
 * @since 1.0.0
 */
Tensor greater(const Tensor& a, const Tensor& b);
```

## Broadcasting

### Broadcasting Rules

TensorCore follows NumPy-style broadcasting rules:

1. **Dimension Alignment**: Dimensions are aligned from the right
2. **Size 1 Broadcasting**: Dimensions of size 1 can broadcast to any size
3. **Missing Dimensions**: Missing dimensions are treated as size 1

### Broadcasting Examples

```cpp
// Example 1: Vector + Matrix
Tensor a = {{1, 2, 3}, {4, 5, 6}};  // Shape: (2, 3)
Tensor b = {10, 20, 30};             // Shape: (3,)
Tensor result = a + b;               // Shape: (2, 3)
// Result: {{11, 22, 33}, {14, 25, 36}}

// Example 2: Scalar + Tensor
Tensor c = {{1, 2}, {3, 4}};        // Shape: (2, 2)
Tensor result2 = c + 10;             // Shape: (2, 2)
// Result: {{11, 12}, {13, 14}}

// Example 3: Complex broadcasting
Tensor d = zeros({3, 1, 4});         // Shape: (3, 1, 4)
Tensor e = zeros({2, 4});            // Shape: (2, 4)
Tensor result3 = d + e;              // Shape: (3, 2, 4)
```

## Performance Considerations

### Memory Efficiency

- **In-place Operations**: Use `+=`, `-=`, `*=`, `/=` when possible to avoid memory allocation
- **Broadcasting**: Broadcasting is memory efficient and doesn't create copies
- **Vectorization**: Operations are vectorized using SIMD instructions when available

### Computational Complexity

- **Element-wise Operations**: O(n) where n is the total number of elements
- **Broadcasting**: O(n) where n is the size of the output tensor
- **Scalar Operations**: O(n) where n is the total number of elements

### Best Practices

```cpp
// Good: Use in-place operations when possible
Tensor a = {1, 2, 3, 4};
Tensor b = {5, 6, 7, 8};
a += b;  // More memory efficient than a = a + b

// Good: Use broadcasting instead of manual expansion
Tensor c = {{1, 2, 3}, {4, 5, 6}};
Tensor d = {10, 20, 30};
Tensor result = c + d;  // Broadcasting is efficient

// Good: Use scalar operations for simple cases
Tensor e = {1, 2, 3, 4};
Tensor result2 = e * 2;  // More efficient than e * ones({4}) * 2
```

## Common Patterns

### Data Normalization

```cpp
// Min-max normalization
Tensor data = {1, 5, 10, 15, 20};
Tensor min_val = data.min();
Tensor max_val = data.max();
Tensor normalized = (data - min_val) / (max_val - min_val);
```

### Masking Operations

```cpp
// Create a mask and apply it
Tensor data = {1, 2, 3, 4, 5};
Tensor mask = data > 3;  // {false, false, false, true, true}
Tensor filtered = data * mask;  // {0, 0, 0, 4, 5}
```

### Weight Updates

```cpp
// Gradient descent update
Tensor weights = {0.1, 0.2, 0.3, 0.4};
Tensor gradients = {0.01, 0.02, 0.03, 0.04};
double learning_rate = 0.1;
weights -= learning_rate * gradients;  // In-place update
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's basic arithmetic operations for their machine learning projects.
