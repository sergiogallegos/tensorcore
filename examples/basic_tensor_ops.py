#!/usr/bin/env python3
"""
Basic Tensor Operations Example

This example demonstrates basic tensor operations using TensorCore.
"""

import tensorcore as tc
import numpy as np

def main():
    print("TensorCore Basic Operations Example")
    print("=" * 40)
    
    # 1. Creating tensors
    print("\n1. Creating Tensors")
    print("-" * 20)
    
    # From list
    a = tc.tensor([1, 2, 3, 4, 5])
    print(f"Tensor from list: {a}")
    
    # From 2D list
    b = tc.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"2D tensor: {b}")
    
    # Using utility functions
    zeros = tc.zeros((3, 3))
    ones = tc.ones((2, 4))
    identity = tc.eye(3)
    
    print(f"Zeros matrix:\n{zeros}")
    print(f"Ones matrix:\n{ones}")
    print(f"Identity matrix:\n{identity}")
    
    # 2. Basic arithmetic operations
    print("\n2. Basic Arithmetic Operations")
    print("-" * 30)
    
    x = tc.tensor([1, 2, 3, 4])
    y = tc.tensor([5, 6, 7, 8])
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x + y = {x + y}")
    print(f"x - y = {x - y}")
    print(f"x * y = {x * y}")
    print(f"x / y = {x / y}")
    print(f"x ** 2 = {x ** 2}")
    
    # 3. Scalar operations
    print("\n3. Scalar Operations")
    print("-" * 20)
    
    print(f"x + 10 = {x + 10}")
    print(f"x * 2 = {x * 2}")
    print(f"10 - x = {10 - x}")
    
    # 4. Matrix operations
    print("\n4. Matrix Operations")
    print("-" * 20)
    
    A = tc.tensor([[1, 2], [3, 4]])
    B = tc.tensor([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"A @ B (matrix multiplication):\n{A.matmul(B)}")
    print(f"A.T (transpose):\n{A.transpose()}")
    
    # 5. Statistical operations
    print("\n5. Statistical Operations")
    print("-" * 25)
    
    data = tc.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Data: {data}")
    print(f"Sum: {data.sum()}")
    print(f"Mean: {data.mean()}")
    print(f"Max: {data.max()}")
    print(f"Min: {data.min()}")
    print(f"Standard deviation: {data.std()}")
    
    # 6. Shape operations
    print("\n6. Shape Operations")
    print("-" * 20)
    
    matrix = tc.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"Original matrix:\n{matrix}")
    print(f"Shape: {matrix.shape}")
    print(f"Number of dimensions: {matrix.ndim}")
    print(f"Total size: {matrix.size}")
    
    # Reshape
    reshaped = matrix.reshape((3, 2))
    print(f"Reshaped to (3, 2):\n{reshaped}")
    
    # 7. Random tensors
    print("\n7. Random Tensors")
    print("-" * 15)
    
    random_normal = tc.random_normal((2, 3), mean=0, std=1)
    random_uniform = tc.random_uniform((2, 3), min=0, max=1)
    
    print(f"Random normal (2x3):\n{random_normal}")
    print(f"Random uniform (2x3):\n{random_uniform}")
    
    # 8. Broadcasting
    print("\n8. Broadcasting")
    print("-" * 15)
    
    a = tc.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    b = tc.tensor([10, 20, 30])             # 3
    
    print(f"Matrix a (2x3):\n{a}")
    print(f"Vector b (3): {b}")
    print(f"a + b (broadcasting):\n{a + b}")
    
    # 9. Comparison with NumPy (if available)
    print("\n9. Comparison with NumPy")
    print("-" * 25)
    
    try:
        import numpy as np
        
        # Create same data in both libraries
        tc_data = tc.tensor([1, 2, 3, 4, 5])
        np_data = np.array([1, 2, 3, 4, 5])
        
        print(f"TensorCore: {tc_data}")
        print(f"NumPy: {np_data}")
        print(f"TensorCore sum: {tc_data.sum()}")
        print(f"NumPy sum: {np_data.sum()}")
        
    except ImportError:
        print("NumPy not available for comparison")
    
    print("\n" + "=" * 40)
    print("Basic operations example completed! 🎉")

if __name__ == "__main__":
    main()
