# Linear Algebra Operations

This document provides comprehensive documentation for all linear algebra operations in TensorCore.

## Table of Contents

1. [Matrix Operations](#matrix-operations)
2. [Vector Operations](#vector-operations)
3. [Matrix Decompositions](#matrix-decompositions)
4. [Matrix Properties](#matrix-properties)
5. [Linear System Solving](#linear-system-solving)
6. [Performance Considerations](#performance-considerations)

## Matrix Operations

### `matmul`

```cpp
/**
 * @brief Performs matrix multiplication using optimized BLAS routines
 * 
 * @details This function computes the matrix product C = A @ B, where:
 * - A is an m×k matrix
 * - B is a k×n matrix  
 * - C is an m×n matrix
 * 
 * The computation is performed as:
 *     C[i,j] = Σ(k=0 to k-1) A[i,k] * B[k,j]
 * 
 * This implementation leverages BLAS (Basic Linear Algebra Subprograms)
 * for optimal performance, particularly for large matrices.
 * 
 * @param a First matrix (m×k)
 * @param b Second matrix (k×n)
 * @return Result matrix (m×n)
 * 
 * @throws ShapeError if inner dimensions don't match (a.cols() != b.rows())
 * 
 * @complexity O(m×k×n) for the computation, but optimized with BLAS
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2}, {3, 4}};  // 2×2
 * Tensor B = {{5, 6}, {7, 8}};  // 2×2
 * Tensor C = matmul(A, B);      // {{19, 22}, {43, 50}}
 * 
 * // Neural network forward pass
 * Tensor input = {{1, 2, 3}, {4, 5, 6}};  // (2, 3)
 * Tensor weights = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};  // (3, 2)
 * Tensor output = matmul(input, weights);  // (2, 2)
 * ```
 * 
 * @see dot, outer, transpose
 * @since 1.0.0
 */
Tensor matmul(const Tensor& a, const Tensor& b);
```

### `dot`

```cpp
/**
 * @brief Computes the dot product of two vectors
 * 
 * @details This function computes the dot product (scalar product) of two
 * vectors. The operation is mathematically defined as:
 * 
 *     result = Σ(i=0 to n-1) a[i] * b[i]
 * 
 * Both input tensors must be 1D vectors of the same length.
 * 
 * @param a First vector
 * @param b Second vector
 * @return Scalar dot product value
 * 
 * @throws ShapeError if tensors are not 1D or have different lengths
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3};
 * Tensor b = {4, 5, 6};
 * double result = dot(a, b);  // 1*4 + 2*5 + 3*6 = 32
 * 
 * // Cosine similarity
 * Tensor v1 = {1, 2, 3};
 * Tensor v2 = {4, 5, 6};
 * double cos_sim = dot(v1, v2) / (norm(v1) * norm(v2));
 * ```
 * 
 * @see matmul, outer, norm
 * @since 1.0.0
 */
Tensor dot(const Tensor& a, const Tensor& b);
```

### `outer`

```cpp
/**
 * @brief Computes the outer product of two vectors
 * 
 * @details This function computes the outer product of two vectors, resulting
 * in a matrix. The operation is mathematically defined as:
 * 
 *     result[i,j] = a[i] * b[j]
 * 
 * If a has length m and b has length n, the result is an m×n matrix.
 * 
 * @param a First vector (length m)
 * @param b Second vector (length n)
 * @return Matrix of shape (m, n)
 * 
 * @throws ShapeError if tensors are not 1D
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 2, 3};
 * Tensor b = {4, 5};
 * Tensor result = outer(a, b);
 * // {{4, 5}, {8, 10}, {12, 15}}
 * 
 * // Rank-1 matrix construction
 * Tensor u = {1, 2, 3};
 * Tensor v = {4, 5, 6};
 * Tensor rank1 = outer(u, v);  // 3×3 rank-1 matrix
 * ```
 * 
 * @see dot, matmul, transpose
 * @since 1.0.0
 */
Tensor outer(const Tensor& a, const Tensor& b);
```

### `cross`

```cpp
/**
 * @brief Computes the cross product of two 3D vectors
 * 
 * @details This function computes the cross product of two 3D vectors.
 * The operation is mathematically defined as:
 * 
 *     result = a × b = [a[1]*b[2] - a[2]*b[1],
 *                       a[2]*b[0] - a[0]*b[2],
 *                       a[0]*b[1] - a[1]*b[0]]
 * 
 * Both input tensors must be 3D vectors.
 * 
 * @param a First 3D vector
 * @param b Second 3D vector
 * @return 3D vector cross product
 * 
 * @throws ShapeError if tensors are not 3D vectors
 * 
 * @example
 * ```cpp
 * Tensor a = {1, 0, 0};
 * Tensor b = {0, 1, 0};
 * Tensor result = cross(a, b);  // {0, 0, 1}
 * 
 * // Normal vector computation
 * Tensor v1 = {1, 2, 3};
 * Tensor v2 = {4, 5, 6};
 * Tensor normal = cross(v1, v2);  // {-3, 6, -3}
 * ```
 * 
 * @see dot, matmul, norm
 * @since 1.0.0
 */
Tensor cross(const Tensor& a, const Tensor& b);
```

### `transpose`

```cpp
/**
 * @brief Transposes a matrix
 * 
 * @details This function transposes a matrix by swapping rows and columns.
 * The operation is mathematically defined as:
 * 
 *     result[i,j] = matrix[j,i]
 * 
 * For a 2D matrix, this swaps the row and column indices.
 * 
 * @param matrix Input matrix
 * @return Transposed matrix
 * 
 * @throws ShapeError if tensor is not 2D
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2, 3}, {4, 5, 6}};  // 2×3
 * Tensor A_T = transpose(A);          // 3×2
 * // {{1, 4}, {2, 5}, {3, 6}}
 * 
 * // Verify transpose property: (A^T)^T = A
 * Tensor A_TT = transpose(transpose(A));
 * // A_TT should equal A
 * ```
 * 
 * @see matmul, dot, outer
 * @since 1.0.0
 */
Tensor transpose(const Tensor& matrix);

/**
 * @brief Transposes a tensor along specified axes
 * 
 * @details This function transposes a tensor by permuting the axes according
 * to the specified permutation. The operation is mathematically defined as:
 * 
 *     result[i0,i1,...,in] = tensor[axes[0],axes[1],...,axes[n]]
 * 
 * @param tensor Input tensor
 * @param axes Permutation of axes
 * @return Transposed tensor
 * 
 * @throws ShapeError if axes are invalid
 * 
 * @example
 * ```cpp
 * Tensor A = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};  // (2, 2, 2)
 * Tensor result = transpose(A, {2, 0, 1});          // (2, 2, 2)
 * // Swaps axes 0 and 2
 * ```
 * 
 * @see transpose(matrix), matmul
 * @since 1.0.0
 */
Tensor transpose(const Tensor& tensor, const std::vector<int>& axes);
```

## Vector Operations

### `norm`

```cpp
/**
 * @brief Computes the vector norm
 * 
 * @details This function computes the p-norm of a vector. The operation is
 * mathematically defined as:
 * 
 *     result = (Σ(i=0 to n-1) |x[i]|^p)^(1/p)
 * 
 * For p=2, this is the Euclidean norm (L2 norm).
 * For p=1, this is the Manhattan norm (L1 norm).
 * For p=∞, this is the maximum norm (L∞ norm).
 * 
 * @param x Input vector
 * @param p Norm order (default: 2)
 * @return Scalar norm value
 * 
 * @throws ShapeError if tensor is not 1D
 * 
 * @example
 * ```cpp
 * Tensor x = {3, 4, 5};
 * double l2_norm = norm(x);        // √(3² + 4² + 5²) = √50 ≈ 7.071
 * double l1_norm = norm(x, 1);     // |3| + |4| + |5| = 12
 * double linf_norm = norm(x, INF); // max(|3|, |4|, |5|) = 5
 * 
 * // Normalize vector
 * Tensor normalized = x / norm(x);
 * ```
 * 
 * @see dot, matmul, transpose
 * @since 1.0.0
 */
Tensor norm(const Tensor& x, double p = 2.0);

/**
 * @brief Computes the matrix norm
 * 
 * @details This function computes the Frobenius norm of a matrix, which is
 * the square root of the sum of squares of all elements.
 * 
 * @param matrix Input matrix
 * @return Scalar Frobenius norm value
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2}, {3, 4}};
 * double frobenius_norm = norm(A);  // √(1² + 2² + 3² + 4²) = √30 ≈ 5.477
 * ```
 * 
 * @see norm(vector), dot, matmul
 * @since 1.0.0
 */
Tensor norm(const Tensor& matrix);
```

## Matrix Decompositions

### `eig`

```cpp
/**
 * @brief Computes eigenvalues and eigenvectors of a square matrix
 * 
 * @details This function computes the eigenvalues and eigenvectors of a
 * square matrix A. The eigenvalues λ and eigenvectors v satisfy:
 * 
 *     A * v = λ * v
 * 
 * The function returns a pair of tensors: (eigenvalues, eigenvectors).
 * 
 * @param matrix Input square matrix
 * @return Pair of (eigenvalues, eigenvectors) tensors
 * 
 * @throws ShapeError if matrix is not square
 * @throws std::runtime_error if eigenvalue computation fails
 * 
 * @example
 * ```cpp
 * Tensor A = {{4, 1}, {1, 3}};  // Symmetric matrix
 * auto [eigenvals, eigenvecs] = eig(A);
 * 
 * // Verify: A * v = λ * v
 * Tensor v = eigenvecs[0];  // First eigenvector
 * Tensor Av = matmul(A, v);
 * Tensor lambda_v = eigenvals[0] * v;
 * // Av should equal lambda_v
 * ```
 * 
 * @see eigh, svd
 * @since 1.0.0
 */
std::pair<Tensor, Tensor> eig(const Tensor& matrix);
```

### `eigh`

```cpp
/**
 * @brief Computes eigenvalues and eigenvectors of a symmetric matrix
 * 
 * @details This function computes the eigenvalues and eigenvectors of a
 * symmetric matrix A. It's more efficient than eig() for symmetric matrices
 * and guarantees real eigenvalues and orthogonal eigenvectors.
 * 
 * @param matrix Input symmetric matrix
 * @return Pair of (eigenvalues, eigenvectors) tensors
 * 
 * @throws ShapeError if matrix is not square
 * @throws std::runtime_error if eigenvalue computation fails
 * 
 * @example
 * ```cpp
 * Tensor A = {{4, 1}, {1, 3}};  // Symmetric matrix
 * auto [eigenvals, eigenvecs] = eigh(A);
 * 
 * // Eigenvalues are real and sorted in descending order
 * // Eigenvectors are orthogonal
 * ```
 * 
 * @see eig, svd
 * @since 1.0.0
 */
std::pair<Tensor, Tensor> eigh(const Tensor& matrix);
```

### `svd`

```cpp
/**
 * @brief Computes the Singular Value Decomposition of a matrix
 * 
 * @details This function computes the SVD of a matrix A, decomposing it as:
 * 
 *     A = U * Σ * V^T
 * 
 * where U and V are orthogonal matrices and Σ is a diagonal matrix of
 * singular values.
 * 
 * @param matrix Input matrix
 * @return Tuple of (U, Σ, V^T) tensors
 * 
 * @throws std::runtime_error if SVD computation fails
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2}, {3, 4}, {5, 6}};  // 3×2 matrix
 * auto [U, S, Vt] = svd(A);
 * 
 * // Verify: A = U * Σ * V^T
 * Tensor Sigma = diag(S);
 * Tensor reconstructed = matmul(matmul(U, Sigma), Vt);
 * // reconstructed should equal A
 * ```
 * 
 * @see eig, eigh
 * @since 1.0.0
 */
std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& matrix);
```

## Matrix Properties

### `trace`

```cpp
/**
 * @brief Computes the trace of a square matrix
 * 
 * @details This function computes the trace of a square matrix, which is
 * the sum of the diagonal elements. The operation is mathematically defined as:
 * 
 *     result = Σ(i=0 to n-1) A[i,i]
 * 
 * @param matrix Input square matrix
 * @return Scalar trace value
 * 
 * @throws ShapeError if matrix is not square
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2}, {3, 4}};
 * double trace_A = trace(A);  // 1 + 4 = 5
 * 
 * // Trace properties
 * Tensor B = {{5, 6}, {7, 8}};
 * double trace_AB = trace(matmul(A, B));
 * double trace_BA = trace(matmul(B, A));
 * // trace_AB should equal trace_BA
 * ```
 * 
 * @see det, inv
 * @since 1.0.0
 */
Tensor trace(const Tensor& matrix);
```

### `det`

```cpp
/**
 * @brief Computes the determinant of a square matrix
 * 
 * @details This function computes the determinant of a square matrix using
 * LU decomposition. The determinant is a scalar value that characterizes
 * the matrix properties.
 * 
 * @param matrix Input square matrix
 * @return Scalar determinant value
 * 
 * @throws ShapeError if matrix is not square
 * @throws std::runtime_error if matrix is singular
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2}, {3, 4}};
 * double det_A = det(A);  // 1*4 - 2*3 = -2
 * 
 * // Determinant properties
 * Tensor B = {{5, 6}, {7, 8}};
 * double det_AB = det(matmul(A, B));
 * double det_A_det_B = det(A) * det(B);
 * // det_AB should equal det_A_det_B
 * ```
 * 
 * @see trace, inv
 * @since 1.0.0
 */
Tensor det(const Tensor& matrix);
```

### `inv`

```cpp
/**
 * @brief Computes the inverse of a square matrix
 * 
 * @details This function computes the inverse of a square matrix using
 * LU decomposition. The inverse A^(-1) satisfies:
 * 
 *     A * A^(-1) = A^(-1) * A = I
 * 
 * where I is the identity matrix.
 * 
 * @param matrix Input square matrix
 * @return Inverse matrix
 * 
 * @throws ShapeError if matrix is not square
 * @throws std::runtime_error if matrix is singular
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2}, {3, 4}};
 * Tensor A_inv = inv(A);
 * 
 * // Verify: A * A^(-1) = I
 * Tensor I = matmul(A, A_inv);
 * // I should be approximately the identity matrix
 * ```
 * 
 * @see det, solve
 * @since 1.0.0
 */
Tensor inv(const Tensor& matrix);
```

### `pinv`

```cpp
/**
 * @brief Computes the Moore-Penrose pseudoinverse of a matrix
 * 
 * @details This function computes the pseudoinverse of a matrix, which is
 * a generalization of the matrix inverse for non-square or singular matrices.
 * It's computed using SVD.
 * 
 * @param matrix Input matrix
 * @return Pseudoinverse matrix
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 2}, {3, 4}, {5, 6}};  // 3×2 matrix
 * Tensor A_pinv = pinv(A);
 * 
 * // Verify: A * A^+ * A = A
 * Tensor reconstructed = matmul(matmul(A, A_pinv), A);
 * // reconstructed should equal A
 * ```
 * 
 * @see inv, svd
 * @since 1.0.0
 */
Tensor pinv(const Tensor& matrix);
```

## Linear System Solving

### `solve`

```cpp
/**
 * @brief Solves a linear system Ax = b
 * 
 * @details This function solves the linear system Ax = b for x, where A is
 * a square matrix and b is a vector or matrix. It uses LU decomposition
 * for efficient solving.
 * 
 * @param A Coefficient matrix (square)
 * @param b Right-hand side vector or matrix
 * @return Solution vector or matrix
 * 
 * @throws ShapeError if A is not square or dimensions don't match
 * @throws std::runtime_error if A is singular
 * 
 * @example
 * ```cpp
 * Tensor A = {{2, 1}, {1, 3}};  // Coefficient matrix
 * Tensor b = {5, 7};            // Right-hand side
 * Tensor x = solve(A, b);       // Solution: {1.6, 1.8}
 * 
 * // Verify: A * x = b
 * Tensor Ax = matmul(A, x);
 * // Ax should equal b
 * ```
 * 
 * @see inv, lstsq
 * @since 1.0.0
 */
Tensor solve(const Tensor& A, const Tensor& b);
```

### `lstsq`

```cpp
/**
 * @brief Solves a least-squares problem
 * 
 * @details This function solves the least-squares problem min ||Ax - b||₂
 * for x, where A is a matrix and b is a vector. It's useful for
 * overdetermined systems.
 * 
 * @param A Coefficient matrix
 * @param b Right-hand side vector
 * @return Solution vector
 * 
 * @example
 * ```cpp
 * Tensor A = {{1, 1}, {1, 2}, {1, 3}};  // 3×2 matrix
 * Tensor b = {2, 3, 4};                 // 3×1 vector
 * Tensor x = lstsq(A, b);               // Least-squares solution
 * 
 * // Verify: minimize ||Ax - b||₂
 * Tensor residual = matmul(A, x) - b;
 * double error = norm(residual);
 * ```
 * 
 * @see solve, pinv
 * @since 1.0.0
 */
Tensor lstsq(const Tensor& A, const Tensor& b);
```

## Performance Considerations

### BLAS Integration

TensorCore leverages BLAS (Basic Linear Algebra Subprograms) for optimal performance:

- **Level 1 BLAS**: Vector operations (dot, norm)
- **Level 2 BLAS**: Matrix-vector operations
- **Level 3 BLAS**: Matrix-matrix operations (matmul)

### Memory Layout

All matrices use **row-major (C-style)** memory layout for BLAS compatibility:

```cpp
// For a 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
// Memory layout: [1, 2, 3, 4, 5, 6]
// Index mapping: [i, j] -> i * cols + j
```

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `matmul` | O(m×k×n) | Optimized with BLAS |
| `dot` | O(n) | Vectorized |
| `eig` | O(n³) | Iterative method |
| `svd` | O(m×n²) | For m×n matrix |
| `det` | O(n³) | LU decomposition |
| `inv` | O(n³) | LU decomposition |

### Best Practices

```cpp
// Good: Use appropriate operations for the task
Tensor A = {{1, 2}, {3, 4}};
Tensor b = {5, 6};

// For square systems
Tensor x = solve(A, b);

// For overdetermined systems
Tensor x_ls = lstsq(A, b);

// Good: Use in-place operations when possible
Tensor C = A;
C += B;  // More efficient than C = A + B

// Good: Use transpose for matrix operations
Tensor A_T = transpose(A);
Tensor result = matmul(A_T, A);  // A^T * A
```

## Common Patterns

### Neural Network Operations

```cpp
// Forward pass
Tensor input = {{1, 2, 3}, {4, 5, 6}};  // (batch_size, input_size)
Tensor weights = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};  // (input_size, output_size)
Tensor output = matmul(input, weights);  // (batch_size, output_size)

// Backward pass
Tensor grad_output = {{0.1, 0.2}, {0.3, 0.4}};  // (batch_size, output_size)
Tensor grad_weights = matmul(transpose(input), grad_output);  // (input_size, output_size)
Tensor grad_input = matmul(grad_output, transpose(weights));  // (batch_size, input_size)
```

### Principal Component Analysis

```cpp
// Center the data
Tensor data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};  // (n_samples, n_features)
Tensor mean = data.mean(0);  // Mean along samples
Tensor centered = data - mean;

// Compute covariance matrix
Tensor cov = matmul(transpose(centered), centered) / (data.shape[0] - 1);

// Compute eigenvalues and eigenvectors
auto [eigenvals, eigenvecs] = eigh(cov);

// Project to principal components
Tensor projected = matmul(centered, eigenvecs);
```

### Linear Regression

```cpp
// Normal equation: θ = (X^T X)^(-1) X^T y
Tensor X = {{1, 1}, {1, 2}, {1, 3}};  // Design matrix with bias
Tensor y = {2, 3, 4};  // Target values

Tensor XTX = matmul(transpose(X), X);
Tensor XTy = matmul(transpose(X), y);
Tensor theta = matmul(inv(XTX), XTy);

// Predictions
Tensor predictions = matmul(X, theta);
```

This comprehensive documentation provides users with all the information they need to effectively use TensorCore's linear algebra operations for their machine learning projects.
