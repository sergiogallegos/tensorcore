# Mathematical Foundations of TensorCore

This document explains the mathematical concepts and algorithms underlying TensorCore, helping you understand what happens under the hood.

## Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [Tensor Operations](#tensor-operations)
3. [Automatic Differentiation](#automatic-differentiation)
4. [Optimization Algorithms](#optimization-algorithms)
5. [Numerical Stability](#numerical-stability)

## Linear Algebra

### Matrix Multiplication

The core of many machine learning operations is matrix multiplication. TensorCore implements efficient matrix multiplication using BLAS (Basic Linear Algebra Subprograms).

#### Algorithm

For matrices A (m×k) and B (k×n), the product C = AB is computed as:

```
C[i,j] = Σ(k=0 to k-1) A[i,k] * B[k,j]
```

#### Implementation Details

```cpp
// Simplified matrix multiplication implementation
Tensor matmul(const Tensor& A, const Tensor& B) {
    // Check dimensions
    assert(A.shape[1] == B.shape[0]);
    
    // Create result tensor
    Tensor C({A.shape[0], B.shape[1]});
    
    // Use BLAS for efficient computation
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A.shape[0], B.shape[1], A.shape[1],
                1.0, A.data(), A.shape[1],
                B.data(), B.shape[1],
                0.0, C.data(), B.shape[1]);
    
    return C;
}
```

### Eigenvalue Decomposition

For a symmetric matrix A, we can find eigenvalues λ and eigenvectors v such that:

```
Av = λv
```

#### Power Iteration Method

```cpp
std::pair<Tensor, Tensor> power_iteration(const Tensor& A, int max_iter=1000) {
    int n = A.shape[0];
    Tensor v = Tensor::random_normal({n});
    v = v / v.norm();
    
    for (int i = 0; i < max_iter; ++i) {
        Tensor Av = A.matmul(v);
        double lambda = v.dot(Av);
        v = Av / Av.norm();
        
        if (converged(v, lambda)) break;
    }
    
    return {lambda, v};
}
```

### Singular Value Decomposition (SVD)

SVD decomposes a matrix A into:

```
A = U Σ V^T
```

Where U and V are orthogonal matrices and Σ is diagonal.

## Tensor Operations

### Broadcasting

Broadcasting allows operations between tensors of different shapes by expanding dimensions as needed.

#### Rules

1. Dimensions are aligned from the right
2. Dimensions of size 1 can be broadcast to any size
3. Missing dimensions are treated as size 1

#### Example

```cpp
// Tensor A: (3, 1, 4)
// Tensor B: (2, 4)
// Result: (3, 2, 4)

Tensor broadcast_add(const Tensor& A, const Tensor& B) {
    // Determine output shape
    auto output_shape = compute_broadcast_shape(A.shape, B.shape);
    
    // Create output tensor
    Tensor result(output_shape);
    
    // Compute strides for efficient iteration
    auto strides_A = compute_broadcast_strides(A.shape, output_shape);
    auto strides_B = compute_broadcast_strides(B.shape, output_shape);
    
    // Perform element-wise addition
    for (size_t i = 0; i < result.size; ++i) {
        auto indices = compute_indices(i, output_shape);
        auto idx_A = compute_broadcast_index(indices, A.shape, strides_A);
        auto idx_B = compute_broadcast_index(indices, B.shape, strides_B);
        result[i] = A[idx_A] + B[idx_B];
    }
    
    return result;
}
```

### Reduction Operations

Reduction operations compute statistics along specified dimensions.

#### Sum Reduction

```cpp
Tensor sum(const Tensor& tensor, int axis) {
    auto new_shape = tensor.shape;
    new_shape.erase(new_shape.begin() + axis);
    
    Tensor result(new_shape);
    
    // Iterate over all elements
    for (size_t i = 0; i < result.size; ++i) {
        double sum_val = 0.0;
        
        // Sum along the specified axis
        for (int j = 0; j < tensor.shape[axis]; ++j) {
            auto indices = compute_indices(i, new_shape);
            indices.insert(indices.begin() + axis, j);
            sum_val += tensor[compute_index(indices, tensor.shape)];
        }
        
        result[i] = sum_val;
    }
    
    return result;
}
```

## Automatic Differentiation

TensorCore implements automatic differentiation using the computational graph approach.

### Computational Graph

Each operation creates nodes in a computational graph that tracks the forward and backward functions.

```cpp
class GraphNode {
public:
    Tensor tensor;
    std::vector<std::shared_ptr<GraphNode>> inputs;
    std::function<Tensor(const std::vector<Tensor>&)> forward_func;
    std::function<std::vector<Tensor>(const Tensor&, const std::vector<Tensor>&)> backward_func;
    Tensor gradient;
};
```

### Forward Pass

The forward pass computes the output of each operation:

```cpp
Tensor add_forward(const std::vector<Tensor>& inputs) {
    return inputs[0] + inputs[1];
}
```

### Backward Pass

The backward pass computes gradients using the chain rule:

```cpp
std::vector<Tensor> add_backward(const Tensor& grad_output, const std::vector<Tensor>& inputs) {
    return {grad_output, grad_output};  // ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
}
```

### Chain Rule Implementation

```cpp
void backward(const std::shared_ptr<GraphNode>& node) {
    // Topological sort of the computational graph
    auto sorted_nodes = topological_sort(node);
    
    // Initialize gradients
    node->gradient = Tensor::ones_like(node->tensor);
    
    // Backward pass
    for (auto& current_node : sorted_nodes) {
        if (current_node->backward_func) {
            auto input_grads = current_node->backward_func(
                current_node->gradient, 
                get_input_tensors(current_node)
            );
            
            // Accumulate gradients
            for (size_t i = 0; i < current_node->inputs.size(); ++i) {
                current_node->inputs[i]->gradient += input_grads[i];
            }
        }
    }
}
```

## Optimization Algorithms

### Stochastic Gradient Descent (SGD)

SGD updates parameters using:

```
θ = θ - α∇θ
```

Where α is the learning rate and ∇θ is the gradient.

```cpp
void SGD::step() {
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i]->gradient) {
            *parameters[i] = *parameters[i] - learning_rate * *parameters[i]->gradient;
        }
    }
}
```

### Adam Optimizer

Adam combines momentum and adaptive learning rates:

```
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
m̂_t = m_t / (1-β₁ᵗ)
v̂_t = v_t / (1-β₂ᵗ)
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

```cpp
void Adam::step() {
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (parameters[i]->gradient) {
            // Update biased first moment estimate
            first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * *parameters[i]->gradient;
            
            // Update biased second moment estimate
            second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * (*parameters[i]->gradient * *parameters[i]->gradient);
            
            // Bias correction
            auto m_hat = first_moments[i] / (1 - std::pow(beta1, step_count + 1));
            auto v_hat = second_moments[i] / (1 - std::pow(beta2, step_count + 1));
            
            // Update parameter
            *parameters[i] = *parameters[i] - learning_rate * m_hat / (v_hat.sqrt() + eps);
        }
    }
    step_count++;
}
```

## Numerical Stability

### Softmax Implementation

The standard softmax formula can be numerically unstable:

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

#### Stable Implementation

```cpp
Tensor softmax(const Tensor& x) {
    // Subtract max for numerical stability
    auto x_max = x.max();
    auto x_shifted = x - x_max;
    
    auto exp_x = x_shifted.exp();
    auto sum_exp = exp_x.sum();
    
    return exp_x / sum_exp;
}
```

### Log-Sum-Exp Trick

For computing log(Σ exp(x_i)):

```cpp
double log_sum_exp(const Tensor& x) {
    auto x_max = x.max();
    auto x_shifted = x - x_max;
    return x_max + (x_shifted.exp()).sum().log();
}
```

### Gradient Clipping

Prevents exploding gradients:

```cpp
void clip_gradients(std::vector<Tensor>& gradients, double max_norm) {
    double total_norm = 0.0;
    for (const auto& grad : gradients) {
        total_norm += grad.norm() * grad.norm();
    }
    total_norm = std::sqrt(total_norm);
    
    if (total_norm > max_norm) {
        double clip_coef = max_norm / total_norm;
        for (auto& grad : gradients) {
            grad = grad * clip_coef;
        }
    }
}
```

## Memory Management

### Memory Layout

Tensors use row-major (C-style) memory layout for compatibility with BLAS:

```cpp
class Tensor {
private:
    std::vector<double> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    
    size_t compute_index(const std::vector<size_t>& indices) const {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            index += indices[i] * strides[i];
        }
        return index;
    }
};
```

### Memory Pool

For efficient memory allocation:

```cpp
class MemoryPool {
private:
    std::vector<std::unique_ptr<double[]>> blocks;
    std::vector<size_t> free_blocks;
    size_t block_size;
    
public:
    double* allocate(size_t size) {
        if (size <= block_size && !free_blocks.empty()) {
            size_t block_idx = free_blocks.back();
            free_blocks.pop_back();
            return blocks[block_idx].get();
        }
        
        // Allocate new block
        blocks.push_back(std::make_unique<double[]>(size));
        return blocks.back().get();
    }
};
```

This mathematical foundation provides the theoretical basis for understanding how TensorCore works internally. The actual implementation includes many optimizations and edge cases not covered here, but these core concepts form the backbone of the library.
