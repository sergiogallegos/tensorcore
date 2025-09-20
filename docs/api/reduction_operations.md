# Reduction Operations

This document provides comprehensive documentation for all reduction operations in TensorCore.

## Table of Contents

1. [Basic Reductions](#basic-reductions)
2. [Statistical Reductions](#statistical-reductions)
3. [Index-based Reductions](#index-based-reductions)
4. [Multi-axis Reductions](#multi-axis-reductions)
5. [Performance Considerations](#performance-considerations)

## Basic Reductions

### `sum`

```cpp
/**
 * @brief Computes the sum of all elements
 * 
 * @details This function computes the sum of all elements in the tensor.
 * The operation is mathematically defined as:
 * 
 *     result = Σ(i=0 to n-1) tensor[i]
 * 
 * @param tensor Input tensor
 * @return Scalar sum value
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4, 5};
 * double result = sum(a);  // 15
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * double result2 = sum(b);  // 21
 * ```
 * 
 * @see mean, max, min, prod
 * @since 1.0.0
 */
Tensor sum(const Tensor& tensor);

/**
 * @brief Computes the sum along a specific axis
 * 
 * @details This function computes the sum along the specified axis, reducing
 * the dimensionality by 1. The operation is mathematically defined as:
 * 
 *     result[i] = Σ(j=0 to shape[axis]-1) tensor[..., j, ...]
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to compute the sum
 * @return Tensor with reduced dimension
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor result = sum(a, 0);          // [5, 7, 9] (sum along rows)
 * Tensor result2 = sum(a, 1);         // [6, 15] (sum along columns)
 * ```
 * 
 * @see sum(tensor), mean, max, min
 * @since 1.0.0
 */
Tensor sum(const Tensor& tensor, int axis);

/**
 * @brief Computes the sum along multiple axes
 * 
 * @details This function computes the sum along multiple axes, reducing
 * the dimensionality by the number of axes specified.
 * 
 * @param tensor Input tensor
 * @param axes Vector of axes along which to compute the sum
 * @return Tensor with reduced dimensions
 * 
 * @throws ShapeError if any axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = sum(a, {0, 1});                   // [16, 20] (sum along first two axes)
 * ```
 * 
 * @see sum(tensor), sum(tensor, axis)
 * @since 1.0.0
 */
Tensor sum(const Tensor& tensor, const std::vector<int>& axes);
```

### `mean`

```cpp
/**
 * @brief Computes the mean of all elements
 * 
 * @details This function computes the arithmetic mean of all elements in the
 * tensor. The operation is mathematically defined as:
 * 
 *     result = (1/n) * Σ(i=0 to n-1) tensor[i]
 * 
 * @param tensor Input tensor
 * @return Scalar mean value
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4, 5};
 * double result = mean(a);  // 3.0
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * double result2 = mean(b);  // 3.5
 * ```
 * 
 * @see sum, max, min, std
 * @since 1.0.0
 */
Tensor mean(const Tensor& tensor);

/**
 * @brief Computes the mean along a specific axis
 * 
 * @details This function computes the arithmetic mean along the specified axis,
 * reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to compute the mean
 * @return Tensor with reduced dimension
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor result = mean(a, 0);         // [2.5, 3.5, 4.5] (mean along rows)
 * Tensor result2 = mean(a, 1);        // [2, 5] (mean along columns)
 * ```
 * 
 * @see mean(tensor), sum, max, min
 * @since 1.0.0
 */
Tensor mean(const Tensor& tensor, int axis);

/**
 * @brief Computes the mean along multiple axes
 * 
 * @details This function computes the arithmetic mean along multiple axes,
 * reducing the dimensionality by the number of axes specified.
 * 
 * @param tensor Input tensor
 * @param axes Vector of axes along which to compute the mean
 * @return Tensor with reduced dimensions
 * 
 * @throws ShapeError if any axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = mean(a, {0, 1});                  // [4, 5] (mean along first two axes)
 * ```
 * 
 * @see mean(tensor), mean(tensor, axis)
 * @since 1.0.0
 */
Tensor mean(const Tensor& tensor, const std::vector<int>& axes);
```

### `max`

```cpp
/**
 * @brief Computes the maximum of all elements
 * 
 * @details This function computes the maximum value among all elements in the
 * tensor. The operation is mathematically defined as:
 * 
 *     result = max(i=0 to n-1) tensor[i]
 * 
 * @param tensor Input tensor
 * @return Scalar maximum value
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 5, 3, 9, 2};
 * double result = max(a);  // 9
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * double result2 = max(b);  // 6
 * ```
 * 
 * @see min, sum, mean, argmax
 * @since 1.0.0
 */
Tensor max(const Tensor& tensor);

/**
 * @brief Computes the maximum along a specific axis
 * 
 * @details This function computes the maximum value along the specified axis,
 * reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to compute the maximum
 * @return Tensor with reduced dimension
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 5, 3}, {4, 2, 6}};  // (2, 3)
 * Tensor result = max(a, 0);          // [4, 5, 6] (max along rows)
 * Tensor result2 = max(a, 1);         // [5, 6] (max along columns)
 * ```
 * 
 * @see max(tensor), min, argmax
 * @since 1.0.0
 */
Tensor max(const Tensor& tensor, int axis);

/**
 * @brief Computes the maximum along multiple axes
 * 
 * @details This function computes the maximum value along multiple axes,
 * reducing the dimensionality by the number of axes specified.
 * 
 * @param tensor Input tensor
 * @param axes Vector of axes along which to compute the maximum
 * @return Tensor with reduced dimensions
 * 
 * @throws ShapeError if any axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 5}, {3, 2}}, {{4, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = max(a, {0, 1});                   // [7, 8] (max along first two axes)
 * ```
 * 
 * @see max(tensor), max(tensor, axis)
 * @since 1.0.0
 */
Tensor max(const Tensor& tensor, const std::vector<int>& axes);
```

### `min`

```cpp
/**
 * @brief Computes the minimum of all elements
 * 
 * @details This function computes the minimum value among all elements in the
 * tensor. The operation is mathematically defined as:
 * 
 *     result = min(i=0 to n-1) tensor[i]
 * 
 * @param tensor Input tensor
 * @return Scalar minimum value
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 5, 3, 9, 2};
 * double result = min(a);  // 1
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * double result2 = min(b);  // 1
 * ```
 * 
 * @see max, sum, mean, argmin
 * @since 1.0.0
 */
Tensor min(const Tensor& tensor);

/**
 * @brief Computes the minimum along a specific axis
 * 
 * @details This function computes the minimum value along the specified axis,
 * reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to compute the minimum
 * @return Tensor with reduced dimension
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 5, 3}, {4, 2, 6}};  // (2, 3)
 * Tensor result = min(a, 0);          // [1, 2, 3] (min along rows)
 * Tensor result2 = min(a, 1);         // [1, 2] (min along columns)
 * ```
 * 
 * @see min(tensor), max, argmin
 * @since 1.0.0
 */
Tensor min(const Tensor& tensor, int axis);

/**
 * @brief Computes the minimum along multiple axes
 * 
 * @details This function computes the minimum value along multiple axes,
 * reducing the dimensionality by the number of axes specified.
 * 
 * @param tensor Input tensor
 * @param axes Vector of axes along which to compute the minimum
 * @return Tensor with reduced dimensions
 * 
 * @throws ShapeError if any axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 5}, {3, 2}}, {{4, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = min(a, {0, 1});                   // [1, 2] (min along first two axes)
 * ```
 * 
 * @see min(tensor), min(tensor, axis)
 * @since 1.0.0
 */
Tensor min(const Tensor& tensor, const std::vector<int>& axes);
```

### `prod`

```cpp
/**
 * @brief Computes the product of all elements
 * 
 * @details This function computes the product of all elements in the tensor.
 * The operation is mathematically defined as:
 * 
 *     result = Π(i=0 to n-1) tensor[i]
 * 
 * @param tensor Input tensor
 * @return Scalar product value
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4};
 * double result = prod(a);  // 24
 * 
 * Tensor b = {{1, 2}, {3, 4}};
 * double result2 = prod(b);  // 24
 * ```
 * 
 * @see sum, mean, max, min
 * @since 1.0.0
 */
Tensor prod(const Tensor& tensor);

/**
 * @brief Computes the product along a specific axis
 * 
 * @details This function computes the product along the specified axis,
 * reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to compute the product
 * @return Tensor with reduced dimension
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor result = prod(a, 0);         // [4, 10, 18] (product along rows)
 * Tensor result2 = prod(a, 1);        // [6, 120] (product along columns)
 * ```
 * 
 * @see prod(tensor), sum, mean
 * @since 1.0.0
 */
Tensor prod(const Tensor& tensor, int axis);

/**
 * @brief Computes the product along multiple axes
 * 
 * @details This function computes the product along multiple axes,
 * reducing the dimensionality by the number of axes specified.
 * 
 * @param tensor Input tensor
 * @param axes Vector of axes along which to compute the product
 * @return Tensor with reduced dimensions
 * 
 * @throws ShapeError if any axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = prod(a, {0, 1});                  // [105, 384] (product along first two axes)
 * ```
 * 
 * @see prod(tensor), prod(tensor, axis)
 * @since 1.0.0
 */
Tensor prod(const Tensor& tensor, const std::vector<int>& axes);
```

## Statistical Reductions

### `std`

```cpp
/**
 * @brief Computes the standard deviation of all elements
 * 
 * @details This function computes the sample standard deviation of all elements
 * in the tensor. The operation is mathematically defined as:
 * 
 *     result = sqrt((1/(n-1)) * Σ(i=0 to n-1) (tensor[i] - mean)²)
 * 
 * @param tensor Input tensor
 * @return Scalar standard deviation value
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4, 5};
 * double result = std(a);  // ≈ 1.581
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * double result2 = std(b);  // ≈ 1.871
 * ```
 * 
 * @see var, mean, sum
 * @since 1.0.0
 */
Tensor std(const Tensor& tensor);

/**
 * @brief Computes the standard deviation along a specific axis
 * 
 * @details This function computes the sample standard deviation along the
 * specified axis, reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to compute the standard deviation
 * @return Tensor with reduced dimension
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor result = std(a, 0);          // [2.121, 2.121, 2.121] (std along rows)
 * Tensor result2 = std(a, 1);         // [1, 1] (std along columns)
 * ```
 * 
 * @see std(tensor), var, mean
 * @since 1.0.0
 */
Tensor std(const Tensor& tensor, int axis);
```

### `var`

```cpp
/**
 * @brief Computes the variance of all elements
 * 
 * @details This function computes the sample variance of all elements in the
 * tensor. The operation is mathematically defined as:
 * 
 *     result = (1/(n-1)) * Σ(i=0 to n-1) (tensor[i] - mean)²
 * 
 * @param tensor Input tensor
 * @return Scalar variance value
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3, 4, 5};
 * double result = var(a);  // ≈ 2.5
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * double result2 = var(b);  // ≈ 3.5
 * ```
 * 
 * @see std, mean, sum
 * @since 1.0.0
 */
Tensor var(const Tensor& tensor);

/**
 * @brief Computes the variance along a specific axis
 * 
 * @details This function computes the sample variance along the specified axis,
 * reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to compute the variance
 * @return Tensor with reduced dimension
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor result = var(a, 0);          // [4.5, 4.5, 4.5] (variance along rows)
 * Tensor result2 = var(a, 1);         // [1, 1] (variance along columns)
 * ```
 * 
 * @see var(tensor), std, mean
 * @since 1.0.0
 */
Tensor var(const Tensor& tensor, int axis);
```

## Index-based Reductions

### `argmax`

```cpp
/**
 * @brief Finds the index of the maximum element
 * 
 * @details This function finds the index of the maximum element in the tensor.
 * If there are multiple maximum values, it returns the index of the first one.
 * 
 * @param tensor Input tensor
 * @return Index of the maximum element
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 5, 3, 9, 2};
 * int result = argmax(a);  // 3 (index of 9)
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * int result2 = argmax(b);  // 5 (index of 6 in flattened array)
 * ```
 * 
 * @see argmin, max, min
 * @since 1.0.0
 */
Tensor argmax(const Tensor& tensor);

/**
 * @brief Finds the index of the maximum element along a specific axis
 * 
 * @details This function finds the index of the maximum element along the
 * specified axis, reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to find the maximum
 * @return Tensor with indices of maximum elements
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 5, 3}, {4, 2, 6}};  // (2, 3)
 * Tensor result = argmax(a, 0);       // [1, 0, 1] (indices along rows)
 * Tensor result2 = argmax(a, 1);      // [1, 2] (indices along columns)
 * ```
 * 
 * @see argmax(tensor), argmin, max
 * @since 1.0.0
 */
Tensor argmax(const Tensor& tensor, int axis);
```

### `argmin`

```cpp
/**
 * @brief Finds the index of the minimum element
 * 
 * @details This function finds the index of the minimum element in the tensor.
 * If there are multiple minimum values, it returns the index of the first one.
 * 
 * @param tensor Input tensor
 * @return Index of the minimum element
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 5, 3, 9, 2};
 * int result = argmin(a);  // 0 (index of 1)
 * 
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};
 * int result2 = argmin(b);  // 0 (index of 1 in flattened array)
 * ```
 * 
 * @see argmax, min, max
 * @since 1.0.0
 */
Tensor argmin(const Tensor& tensor);

/**
 * @brief Finds the index of the minimum element along a specific axis
 * 
 * @details This function finds the index of the minimum element along the
 * specified axis, reducing the dimensionality by 1.
 * 
 * @param tensor Input tensor
 * @param axis Axis along which to find the minimum
 * @return Tensor with indices of minimum elements
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 5, 3}, {4, 2, 6}};  // (2, 3)
 * Tensor result = argmin(a, 0);       // [0, 1, 0] (indices along rows)
 * Tensor result2 = argmin(a, 1);      // [0, 1] (indices along columns)
 * ```
 * 
 * @see argmin(tensor), argmax, min
 * @since 1.0.0
 */
Tensor argmin(const Tensor& tensor, int axis);
```

## Multi-axis Reductions

### `sum` with multiple axes

```cpp
/**
 * @brief Computes the sum along multiple axes
 * 
 * @details This function computes the sum along multiple axes, reducing
 * the dimensionality by the number of axes specified. The axes are processed
 * in the order they appear in the vector.
 * 
 * @param tensor Input tensor
 * @param axes Vector of axes along which to compute the sum
 * @return Tensor with reduced dimensions
 * 
 * @throws ShapeError if any axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = sum(a, {0, 1});                   // [16, 20] (sum along first two axes)
 * 
 * // Equivalent to:
 * Tensor temp = sum(a, 0);  // Sum along axis 0
 * Tensor result2 = sum(temp, 0);  // Sum along axis 0 of result
 * ```
 * 
 * @see sum(tensor), sum(tensor, axis)
 * @since 1.0.0
 */
Tensor sum(const Tensor& tensor, const std::vector<int>& axes);
```

### `mean` with multiple axes

```cpp
/**
 * @brief Computes the mean along multiple axes
 * 
 * @details This function computes the arithmetic mean along multiple axes,
 * reducing the dimensionality by the number of axes specified.
 * 
 * @param tensor Input tensor
 * @param axes Vector of axes along which to compute the mean
 * @return Tensor with reduced dimensions
 * 
 * @throws ShapeError if any axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = mean(a, {0, 1});                  // [4, 5] (mean along first two axes)
 * ```
 * 
 * @see mean(tensor), mean(tensor, axis)
 * @since 1.0.0
 */
Tensor mean(const Tensor& tensor, const std::vector<int>& axes);
```

## Performance Considerations

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `sum` | O(n) | Single pass through data |
| `mean` | O(n) | Single pass through data |
| `max/min` | O(n) | Single pass through data |
| `std/var` | O(n) | Two passes (mean + variance) |
| `argmax/argmin` | O(n) | Single pass through data |
| `prod` | O(n) | Single pass through data |

### Memory Efficiency

- **In-place Operations**: Some reductions can be computed in-place
- **Vectorization**: Operations are vectorized using SIMD instructions
- **Cache Efficiency**: Operations are optimized for cache locality

### Best Practices

```cpp
// Good: Use appropriate reduction for the task
Tensor data = {1, 2, 3, 4, 5};

// For statistics
double mean_val = mean(data);
double std_val = std(data);

// For finding extremes
double max_val = max(data);
int max_idx = argmax(data);

// Good: Use axis-specific reductions when possible
Tensor matrix = {{1, 2, 3}, {4, 5, 6}};

// Sum along rows (axis 0)
Tensor row_sums = sum(matrix, 0);

// Sum along columns (axis 1)
Tensor col_sums = sum(matrix, 1);

// Good: Use multi-axis reductions for complex operations
Tensor tensor_3d = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};

// Sum along first two axes
Tensor result = sum(tensor_3d, {0, 1});
```

## Common Patterns

### Data Normalization

```cpp
// Z-score normalization
Tensor data = {1, 2, 3, 4, 5};
Tensor mean_val = mean(data);
Tensor std_val = std(data);
Tensor normalized = (data - mean_val) / std_val;

// Min-max normalization
Tensor min_val = min(data);
Tensor max_val = max(data);
Tensor normalized2 = (data - min_val) / (max_val - min_val);
```

### Loss Functions

```cpp
// Mean Squared Error
Tensor predictions = {1, 2, 3, 4};
Tensor targets = {1.1, 1.9, 3.1, 3.9};
Tensor mse = ((predictions - targets) * (predictions - targets)).mean();

// Cross-Entropy Loss
Tensor logits = {1, 2, 3};
Tensor softmax = exp(logits) / exp(logits).sum();
Tensor ce_loss = -log(softmax).sum();
```

### Statistical Analysis

```cpp
// Compute statistics for each feature
Tensor data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};  // (3, 3)
Tensor feature_means = mean(data, 0);  // Mean of each feature
Tensor feature_stds = std(data, 0);    // Std of each feature

// Find the most important feature
int most_important = argmax(feature_stds);

// Compute correlation matrix
Tensor centered = data - feature_means;
Tensor cov = matmul(transpose(centered), centered) / (data.shape[0] - 1);
```

### Neural Network Operations

```cpp
// Batch normalization
Tensor batch = {{1, 2, 3}, {4, 5, 6}};  // (batch_size, features)
Tensor batch_mean = mean(batch, 0);     // Mean across batch
Tensor batch_var = var(batch, 0);       // Variance across batch
Tensor normalized = (batch - batch_mean) / sqrt(batch_var + 1e-8);

// Global average pooling
Tensor feature_maps = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
Tensor pooled = mean(feature_maps, {1, 2});  // (2,) - average over spatial dimensions
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's reduction operations for their machine learning projects.
