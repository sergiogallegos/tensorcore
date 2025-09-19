# Getting Started with TensorCore

Welcome to TensorCore! This tutorial will guide you through the basics of using TensorCore for machine learning and numerical computing.

## Table of Contents

1. [Installation](#installation)
2. [Basic Tensor Operations](#basic-tensor-operations)
3. [Mathematical Functions](#mathematical-functions)
4. [Linear Algebra](#linear-algebra)
5. [Neural Networks](#neural-networks)
6. [Next Steps](#next-steps)

## Installation

### Prerequisites

Before installing TensorCore, make sure you have:

- C++17 or later
- CMake 3.15+
- Python 3.8+
- BLAS library (OpenBLAS, Intel MKL, or ATLAS)
- pybind11

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/tensorcore.git
cd tensorcore

# Build the library
./scripts/build.sh

# Install Python bindings
cd python
pip install -e .
```

### Quick Test

```python
import tensorcore as tc

# Create a simple tensor
a = tc.tensor([1, 2, 3, 4])
print(f"Tensor: {a}")
print(f"Sum: {a.sum()}")
```

## Basic Tensor Operations

### Creating Tensors

TensorCore provides several ways to create tensors:

```python
import tensorcore as tc

# From Python lists
a = tc.tensor([1, 2, 3, 4])
b = tc.tensor([[1, 2], [3, 4]])

# Using utility functions
zeros = tc.zeros((3, 3))
ones = tc.ones((2, 4))
identity = tc.eye(3)

# Random tensors
random_normal = tc.random_normal((2, 3), mean=0, std=1)
random_uniform = tc.random_uniform((2, 3), min=0, max=1)

# Ranges
range_tensor = tc.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_tensor = tc.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]
```

### Basic Arithmetic

```python
# Element-wise operations
a = tc.tensor([1, 2, 3, 4])
b = tc.tensor([5, 6, 7, 8])

print(a + b)  # [6, 8, 10, 12]
print(a - b)  # [-4, -4, -4, -4]
print(a * b)  # [5, 12, 21, 32]
print(a / b)  # [0.2, 0.333, 0.429, 0.5]

# Scalar operations
print(a + 10)  # [11, 12, 13, 14]
print(a * 2)   # [2, 4, 6, 8]
print(10 - a)  # [9, 8, 7, 6]
```

### Shape Operations

```python
# Reshaping
matrix = tc.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original shape: {matrix.shape}")  # (2, 3)

reshaped = matrix.reshape((3, 2))
print(f"Reshaped: {reshaped.shape}")  # (3, 2)

# Transposing
transposed = matrix.transpose()
print(f"Transposed shape: {transposed.shape}")  # (3, 2)

# Squeezing and unsqueezing
vector = tc.tensor([1, 2, 3])
unsqueezed = vector.unsqueeze(0)  # Shape: (1, 3)
squeezed = unsqueezed.squeeze(0)  # Shape: (3,)
```

## Mathematical Functions

### Element-wise Functions

```python
x = tc.tensor([-2, -1, 0, 1, 2])

# Trigonometric functions
print(tc.sin(x))
print(tc.cos(x))
print(tc.tan(x))

# Exponential and logarithmic functions
print(tc.exp(x))
print(tc.log(tc.abs(x) + 1))  # log(1 + |x|) to avoid log(0)

# Power functions
print(tc.sqrt(tc.abs(x)))
print(tc.pow(x, 2))

# Rounding functions
print(tc.floor(x))
print(tc.ceil(x))
print(tc.round(x))
```

### Reduction Operations

```python
matrix = tc.tensor([[1, 2, 3], [4, 5, 6]])

# Global reductions
print(f"Sum: {matrix.sum()}")        # 21
print(f"Mean: {matrix.mean()}")      # 3.5
print(f"Max: {matrix.max()}")        # 6
print(f"Min: {matrix.min()}")        # 1

# Axis-wise reductions
print(f"Sum along axis 0: {matrix.sum(0)}")  # [5, 7, 9]
print(f"Sum along axis 1: {matrix.sum(1)}")  # [6, 15]
print(f"Mean along axis 0: {matrix.mean(0)}")  # [2.5, 3.5, 4.5]
```

## Linear Algebra

### Matrix Operations

```python
# Matrix multiplication
A = tc.tensor([[1, 2], [3, 4]])
B = tc.tensor([[5, 6], [7, 8]])

C = A.matmul(B)
print(f"A @ B = \n{C}")

# Dot product
a = tc.tensor([1, 2, 3])
b = tc.tensor([4, 5, 6])
dot_product = a.dot(b)
print(f"Dot product: {dot_product}")

# Outer product
outer = a.outer(b)
print(f"Outer product:\n{outer}")
```

### Matrix Properties

```python
# Determinant
A = tc.tensor([[2, 1], [3, 4]])
det_A = A.det()
print(f"Determinant: {det_A}")

# Trace
trace_A = A.trace()
print(f"Trace: {trace_A}")

# Transpose
A_T = A.transpose()
print(f"Transpose:\n{A_T}")

# Matrix norm
norm_A = A.norm()
print(f"Frobenius norm: {norm_A}")
```

### Eigenvalue Decomposition

```python
# Symmetric matrix for eigenvalue decomposition
A = tc.tensor([[4, 1], [1, 3]])

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = tc.eigh(A)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")
```

## Neural Networks

### Building a Simple Network

```python
from tensorcore.nn import Dense, ReLU, Sigmoid, Sequential
from tensorcore.nn import MSELoss
from tensorcore.optim import Adam

# Create a simple neural network
model = Sequential([
    Dense(2, 10, activation=ReLU),
    Dense(10, 5, activation=ReLU),
    Dense(5, 1, activation=Sigmoid)
])

# Loss function and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters, lr=0.01)

# Example forward pass
x = tc.tensor([[1, 2], [3, 4]])
output = model.forward(x)
print(f"Model output: {output}")
```

### Training a Model

```python
# Generate synthetic data
X = tc.random_normal((100, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(int).float()

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model.forward(X)
    loss = criterion.forward(predictions, y)
    
    # Backward pass
    grad_output = criterion.backward(predictions, y)
    model.backward(grad_output)
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

## Next Steps

Now that you've learned the basics of TensorCore, here are some next steps:

1. **Explore Examples**: Check out the `examples/` directory for more comprehensive examples
2. **Read Documentation**: Browse the API documentation in `docs/api/`
3. **Learn Internals**: Understand how TensorCore works under the hood in `docs/internals/`
4. **Contribute**: Help improve TensorCore by contributing code or documentation

### Recommended Learning Path

1. **Basic Operations**: Master tensor creation, arithmetic, and shape operations
2. **Mathematical Functions**: Learn about element-wise functions and reductions
3. **Linear Algebra**: Understand matrix operations and decompositions
4. **Neural Networks**: Build and train simple models
5. **Advanced Topics**: Explore automatic differentiation, optimization, and performance

### Additional Resources

- [API Reference](api/)
- [Mathematical Foundations](internals/mathematics.md)
- [Performance Optimization](internals/performance.md)
- [Memory Management](internals/memory.md)

Happy learning with TensorCore! 🚀
