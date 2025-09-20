# Shape Operations

This document provides comprehensive documentation for all shape operations in TensorCore.

## Table of Contents

1. [Basic Shape Operations](#basic-shape-operations)
2. [Dimension Manipulation](#dimension-manipulation)
3. [Indexing and Slicing](#indexing-and-slicing)
4. [Concatenation and Stacking](#concatenation-and-stacking)
5. [Broadcasting](#broadcasting)
6. [Performance Considerations](#performance-considerations)

## Basic Shape Operations

### `reshape`

```cpp
/**
 * @brief Reshapes a tensor to a new shape
 * 
 * @details This function reshapes a tensor to a new shape while preserving
 * all elements. The total number of elements must remain the same.
 * The operation is mathematically defined as:
 * 
 *     result[i] = tensor[flattened_index(i, new_shape)]
 * 
 * where flattened_index converts multi-dimensional indices to a flat index.
 * 
 * @param new_shape The desired new shape
 * @return Reshaped tensor with the same data
 * 
 * @throws ShapeError if total number of elements doesn't match
 * @throws std::invalid_argument if new_shape is empty or contains zero dimensions
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor b = a.reshape({3, 2});       // (3, 2)
 * // b = {{1, 2}, {3, 4}, {5, 6}}
 * 
 * Tensor c = a.reshape({6});          // (6,)
 * // c = {1, 2, 3, 4, 5, 6}
 * 
 * // Flatten to 1D
 * Tensor flattened = a.reshape({-1}); // -1 means infer from other dimensions
 * ```
 * 
 * @see transpose, squeeze, unsqueeze
 * @since 1.0.0
 */
Tensor reshape(const shape_type& new_shape) const;
```

### `transpose`

```cpp
/**
 * @brief Transposes a 2D matrix
 * 
 * @details This function transposes a 2D matrix by swapping rows and columns.
 * The operation is mathematically defined as:
 * 
 *     result[i,j] = matrix[j,i]
 * 
 * @return Transposed matrix
 * 
 * @throws ShapeError if tensor is not 2D
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor b = a.transpose();            // (3, 2)
 * // b = {{1, 4}, {2, 5}, {3, 6}}
 * 
 * // Verify: (A^T)^T = A
 * Tensor c = b.transpose();            // (2, 3)
 * // c should equal a
 * ```
 * 
 * @see transpose(axes), reshape, matmul
 * @since 1.0.0
 */
Tensor transpose() const;

/**
 * @brief Transposes a tensor along specified axes
 * 
 * @details This function transposes a tensor by permuting the axes according
 * to the specified permutation. The operation is mathematically defined as:
 * 
 *     result[i0,i1,...,in] = tensor[axes[0],axes[1],...,axes[n]]
 * 
 * @param axes Permutation of axes
 * @return Transposed tensor
 * 
 * @throws ShapeError if axes are invalid or out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor b = a.transpose({2, 0, 1});                // (2, 2, 2)
 * // Swaps axes 0 and 2
 * 
 * // Common transpose patterns
 * Tensor c = a.transpose({1, 0, 2});  // Swap first two axes
 * Tensor d = a.transpose({0, 2, 1});  // Swap last two axes
 * ```
 * 
 * @see transpose(), reshape, matmul
 * @since 1.0.0
 */
Tensor transpose(const std::vector<int>& axes) const;
```

### `squeeze`

```cpp
/**
 * @brief Removes dimensions of size 1
 * 
 * @details This function removes all dimensions of size 1 from the tensor.
 * This is useful for cleaning up tensors after operations that may introduce
 * singleton dimensions.
 * 
 * @return Tensor with singleton dimensions removed
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2, 3}}};  // (1, 1, 3)
 * Tensor b = a.squeeze();    // (3,)
 * // b = {1, 2, 3}
 * 
 * Tensor c = {{{1}, {2}, {3}}};  // (1, 3, 1)
 * Tensor d = c.squeeze();        // (3,)
 * // d = {1, 2, 3}
 * ```
 * 
 * @see squeeze(axis), unsqueeze
 * @since 1.0.0
 */
Tensor squeeze() const;

/**
 * @brief Removes a specific dimension of size 1
 * 
 * @details This function removes the specified dimension if it has size 1.
 * If the dimension doesn't have size 1, the tensor is returned unchanged.
 * 
 * @param axis The axis to remove
 * @return Tensor with the specified dimension removed
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{{1, 2, 3}}};  // (1, 1, 3)
 * Tensor b = a.squeeze(0);   // (1, 3) - remove first dimension
 * Tensor c = a.squeeze(1);   // (1, 3) - remove second dimension
 * Tensor d = a.squeeze(2);   // (1, 1, 3) - third dimension has size 3, not removed
 * ```
 * 
 * @see squeeze(), unsqueeze
 * @since 1.0.0
 */
Tensor squeeze(int axis) const;
```

### `unsqueeze`

```cpp
/**
 * @brief Adds a new dimension of size 1
 * 
 * @details This function adds a new dimension of size 1 at the specified
 * position. This is useful for broadcasting operations and maintaining
 * consistent tensor shapes.
 * 
 * @param axis The position where to insert the new dimension
 * @return Tensor with a new dimension of size 1
 * 
 * @throws ShapeError if axis is out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3};      // (3,)
 * Tensor b = a.unsqueeze(0); // (1, 3)
 * // b = {{1, 2, 3}}
 * 
 * Tensor c = a.unsqueeze(1); // (3, 1)
 * // c = {{1}, {2}, {3}}
 * 
 * Tensor d = a.unsqueeze(-1); // (3, 1) - negative indexing from the end
 * ```
 * 
 * @see squeeze, reshape
 * @since 1.0.0
 */
Tensor unsqueeze(int axis) const;
```

## Dimension Manipulation

### `broadcast_to`

```cpp
/**
 * @brief Broadcasts a tensor to a target shape
 * 
 * @details This function broadcasts a tensor to a target shape following
 * NumPy-style broadcasting rules. The tensor is expanded along dimensions
 * of size 1 to match the target shape.
 * 
 * @param target_shape The desired target shape
 * @return Broadcasted tensor
 * 
 * @throws ShapeError if broadcasting is not possible
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3};                    // (3,)
 * Tensor b = a.broadcast_to({2, 3});       // (2, 3)
 * // b = {{1, 2, 3}, {1, 2, 3}}
 * 
 * Tensor c = {{1}, {2}};                   // (2, 1)
 * Tensor d = c.broadcast_to({2, 3});       // (2, 3)
 * // d = {{1, 1, 1}, {2, 2, 2}}
 * ```
 * 
 * @see is_broadcastable, reshape
 * @since 1.0.0
 */
Tensor broadcast_to(const shape_type& target_shape) const;
```

### `is_broadcastable`

```cpp
/**
 * @brief Checks if a tensor can be broadcast to another shape
 * 
 * @details This function checks if the current tensor can be broadcast to
 * the specified shape following NumPy-style broadcasting rules.
 * 
 * @param other The tensor to check broadcasting compatibility with
 * @return True if broadcasting is possible, false otherwise
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3};                    // (3,)
 * Tensor b = {{1, 2, 3}, {4, 5, 6}};       // (2, 3)
 * bool can_broadcast = a.is_broadcastable(b);  // true
 * 
 * Tensor c = {1, 2};                       // (2,)
 * bool cannot_broadcast = c.is_broadcastable(b);  // false
 * ```
 * 
 * @see broadcast_to, reshape
 * @since 1.0.0
 */
bool is_broadcastable(const Tensor& other) const;
```

## Indexing and Slicing

### `slice`

```cpp
/**
 * @brief Extracts a slice from a tensor
 * 
 * @details This function extracts a slice from a tensor using the specified
 * ranges. Each range is a pair of (start, end) indices, where start is
 * inclusive and end is exclusive.
 * 
 * @param ranges Vector of ranges for each dimension
 * @return Sliced tensor
 * 
 * @throws ShapeError if ranges are invalid
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};  // (3, 3)
 * Tensor b = a.slice({{1, 3}, {0, 2}});          // (2, 2)
 * // b = {{4, 5}, {7, 8}}
 * 
 * // Extract first row
 * Tensor c = a.slice({{0, 1}, {0, 3}});          // (1, 3)
 * // c = {{1, 2, 3}}
 * 
 * // Extract last column
 * Tensor d = a.slice({{0, 3}, {2, 3}});          // (3, 1)
 * // d = {{3}, {6}, {9}}
 * ```
 * 
 * @see index, reshape
 * @since 1.0.0
 */
Tensor slice(const std::vector<std::pair<size_type, size_type>>& ranges) const;
```

### `index`

```cpp
/**
 * @brief Extracts elements at specific indices
 * 
 * @details This function extracts elements at the specified indices along
 * each dimension. This is useful for advanced indexing operations.
 * 
 * @param indices Vector of indices for each dimension
 * @return Tensor with elements at specified indices
 * 
 * @throws ShapeError if indices are out of bounds
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};  // (3, 3)
 * Tensor b = a.index({1, 2});                     // Scalar
 * // b = 6 (element at row 1, column 2)
 * 
 * // Extract specific elements
 * Tensor c = a.index({0, 0});                     // Scalar
 * // c = 1
 * ```
 * 
 * @see slice, reshape
 * @since 1.0.0
 */
Tensor index(const std::vector<size_type>& indices) const;
```

## Concatenation and Stacking

### `concatenate`

```cpp
/**
 * @brief Concatenates tensors along a specified axis
 * 
 * @details This function concatenates a list of tensors along the specified
 * axis. All tensors must have the same shape except along the concatenation
 * axis.
 * 
 * @param tensors Vector of tensors to concatenate
 * @param axis Axis along which to concatenate (default: 0)
 * @return Concatenated tensor
 * 
 * @throws ShapeError if tensors have incompatible shapes
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2}, {3, 4}};                    // (2, 2)
 * Tensor b = {{5, 6}, {7, 8}};                    // (2, 2)
 * Tensor c = concatenate({a, b}, 0);              // (4, 2)
 * // c = {{1, 2}, {3, 4}, {5, 6}, {7, 8}}
 * 
 * Tensor d = concatenate({a, b}, 1);              // (2, 4)
 * // d = {{1, 2, 5, 6}, {3, 4, 7, 8}}
 * ```
 * 
 * @see stack, split
 * @since 1.0.0
 */
Tensor concatenate(const std::vector<Tensor>& tensors, int axis = 0);
```

### `stack`

```cpp
/**
 * @brief Stacks tensors along a new axis
 * 
 * @details This function stacks a list of tensors along a new axis. All
 * tensors must have the same shape. The new axis is inserted at the
 * specified position.
 * 
 * @param tensors Vector of tensors to stack
 * @param axis Position where to insert the new axis (default: 0)
 * @return Stacked tensor
 * 
 * @throws ShapeError if tensors have different shapes
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3};                          // (3,)
 * Tensor b = {4, 5, 6};                          // (3,)
 * Tensor c = stack({a, b}, 0);                   // (2, 3)
 * // c = {{1, 2, 3}, {4, 5, 6}}
 * 
 * Tensor d = stack({a, b}, 1);                   // (3, 2)
 * // d = {{1, 4}, {2, 5}, {3, 6}}
 * ```
 * 
 * @see concatenate, split
 * @since 1.0.0
 */
Tensor stack(const std::vector<Tensor>& tensors, int axis = 0);
```

### `split`

```cpp
/**
 * @brief Splits a tensor along a specified axis
 * 
 * @details This function splits a tensor along the specified axis into
 * multiple tensors. The sizes parameter specifies how many elements each
 * resulting tensor should have along the split axis.
 * 
 * @param tensor Input tensor to split
 * @param axis Axis along which to split
 * @param sizes Vector of sizes for each resulting tensor
 * @return Vector of split tensors
 * 
 * @throws ShapeError if sizes don't sum to the axis size
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};  // (3, 3)
 * auto result = split(a, 0, {1, 2});              // Split along axis 0
 * // result[0] = {{1, 2, 3}}                      // (1, 3)
 * // result[1] = {{4, 5, 6}, {7, 8, 9}}          // (2, 3)
 * ```
 * 
 * @see concatenate, stack
 * @since 1.0.0
 */
std::vector<Tensor> split(const Tensor& tensor, int axis, const std::vector<int>& sizes);
```

### `tile`

```cpp
/**
 * @brief Repeats a tensor along specified axes
 * 
 * @details This function repeats a tensor along specified axes. The reps
 * parameter specifies how many times to repeat along each axis.
 * 
 * @param tensor Input tensor to tile
 * @param reps Vector of repetition counts for each axis
 * @return Tiled tensor
 * 
 * @throws ShapeError if reps has wrong length
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2};                             // (2,)
 * Tensor b = tile(a, {2, 3});                    // (4, 6)
 * // b = {{1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2}}
 * ```
 * 
 * @see repeat, reshape
 * @since 1.0.0
 */
Tensor tile(const Tensor& tensor, const std::vector<int>& reps);
```

### `repeat`

```cpp
/**
 * @brief Repeats elements along specified axes
 * 
 * @details This function repeats elements along specified axes. Unlike tile(),
 * this function repeats individual elements rather than entire sub-tensors.
 * 
 * @param tensor Input tensor to repeat
 * @param reps Vector of repetition counts for each axis
 * @return Repeated tensor
 * 
 * @throws ShapeError if reps has wrong length
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2};                             // (2,)
 * Tensor b = repeat(a, {2, 3});                  // (4, 6)
 * // b = {{1, 1, 1, 2, 2, 2}, {1, 1, 1, 2, 2, 2}, {1, 1, 1, 2, 2, 2}, {1, 1, 1, 2, 2, 2}}
 * ```
 * 
 * @see tile, reshape
 * @since 1.0.0
 */
Tensor repeat(const Tensor& tensor, const std::vector<int>& reps);
```

### `pad`

```cpp
/**
 * @brief Pads a tensor with constant values
 * 
 * @details This function pads a tensor with constant values along specified
 * axes. The padding parameter specifies how much to pad on each side of
 * each axis.
 * 
 * @param tensor Input tensor to pad
 * @param padding Vector of (before, after) padding for each axis
 * @param value Value to use for padding (default: 0.0)
 * @return Padded tensor
 * 
 * @throws ShapeError if padding has wrong length
 * 
 * @example
 * ```cpp
 * Tensor a = {{1, 2}, {3, 4}};                   // (2, 2)
 * Tensor b = pad(a, {{1, 1}, {1, 1}}, 0);        // (4, 4)
 * // b = {{0, 0, 0, 0}, {0, 1, 2, 0}, {0, 3, 4, 0}, {0, 0, 0, 0}}
 * ```
 * 
 * @see reshape, slice
 * @since 1.0.0
 */
Tensor pad(const Tensor& tensor, const std::vector<std::pair<int, int>>& padding, double value = 0.0);
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

- **View Operations**: Some operations like `transpose()` create views rather than copies
- **In-place Operations**: Some operations can be performed in-place
- **Lazy Evaluation**: Some operations are deferred until actually needed

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `reshape` | O(1) | View operation, no data movement |
| `transpose` | O(1) | View operation, no data movement |
| `squeeze` | O(1) | View operation, no data movement |
| `unsqueeze` | O(1) | View operation, no data movement |
| `slice` | O(n) | Creates new tensor with copied data |
| `concatenate` | O(n) | Creates new tensor with copied data |
| `stack` | O(n) | Creates new tensor with copied data |

### Best Practices

```cpp
// Good: Use view operations when possible
Tensor a = {{1, 2, 3}, {4, 5, 6}};
Tensor b = a.transpose();  // View operation, no copy

// Good: Use appropriate operations for the task
Tensor c = {1, 2, 3};
Tensor d = c.unsqueeze(0);  // Add batch dimension

// Good: Use broadcasting instead of manual expansion
Tensor e = {{1, 2, 3}, {4, 5, 6}};
Tensor f = {10, 20, 30};
Tensor result = e + f;  // Broadcasting is efficient

// Good: Use in-place operations when possible
Tensor g = {{1, 2}, {3, 4}};
g += 10;  // In-place operation
```

## Common Patterns

### Data Preprocessing

```cpp
// Add batch dimension
Tensor data = {1, 2, 3, 4, 5};
Tensor batch_data = data.unsqueeze(0);  // (1, 5)

// Flatten for neural network input
Tensor image = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
Tensor flattened = image.reshape({-1}); // (6,)

// Normalize along features
Tensor features = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
Tensor mean = features.mean(0);             // (3,)
Tensor normalized = features - mean;        // Broadcasting
```

### Neural Network Operations

```cpp
// Reshape for convolutional layers
Tensor input = {1, 2, 3, 4, 5, 6, 7, 8};  // (8,)
Tensor reshaped = input.reshape({1, 1, 2, 4});  // (1, 1, 2, 4) - (batch, channels, height, width)

// Transpose for matrix multiplication
Tensor weights = {{1, 2}, {3, 4}, {5, 6}};  // (3, 2)
Tensor input = {{1, 2, 3}};                  // (1, 3)
Tensor output = matmul(input, weights.transpose());  // (1, 2)

// Concatenate features
Tensor feat1 = {{1, 2}, {3, 4}};  // (2, 2)
Tensor feat2 = {{5, 6}, {7, 8}};  // (2, 2)
Tensor combined = concatenate({feat1, feat2}, 1);  // (2, 4)
```

### Data Augmentation

```cpp
// Pad images
Tensor image = {{1, 2}, {3, 4}};  // (2, 2)
Tensor padded = pad(image, {{1, 1}, {1, 1}}, 0);  // (4, 4)

// Tile for data augmentation
Tensor sample = {1, 2, 3};  // (3,)
Tensor augmented = tile(sample, {4, 1});  // (12,) - repeat 4 times

// Stack multiple samples
Tensor sample1 = {1, 2, 3};
Tensor sample2 = {4, 5, 6};
Tensor batch = stack({sample1, sample2}, 0);  // (2, 3)
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's shape operations for their machine learning projects.
