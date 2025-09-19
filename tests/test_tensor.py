#!/usr/bin/env python3
"""
Test suite for TensorCore tensor operations

This module contains comprehensive tests for the TensorCore library.
"""

import unittest
import tensorcore as tc
import numpy as np
import math

class TestTensorCreation(unittest.TestCase):
    """Test tensor creation functions."""
    
    def test_tensor_from_list(self):
        """Test creating tensor from Python list."""
        data = [1, 2, 3, 4, 5]
        tensor = tc.tensor(data)
        self.assertEqual(tensor.shape, (5,))
        self.assertEqual(tensor.size, 5)
        self.assertEqual(tensor.data, data)
    
    def test_tensor_from_2d_list(self):
        """Test creating 2D tensor from nested list."""
        data = [[1, 2, 3], [4, 5, 6]]
        tensor = tc.tensor(data)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.size, 6)
    
    def test_zeros(self):
        """Test creating tensor of zeros."""
        tensor = tc.zeros((3, 4))
        self.assertEqual(tensor.shape, (3, 4))
        self.assertTrue(all(x == 0.0 for x in tensor.data))
    
    def test_ones(self):
        """Test creating tensor of ones."""
        tensor = tc.ones((2, 3))
        self.assertEqual(tensor.shape, (2, 3))
        self.assertTrue(all(x == 1.0 for x in tensor.data))
    
    def test_eye(self):
        """Test creating identity matrix."""
        tensor = tc.eye(3)
        self.assertEqual(tensor.shape, (3, 3))
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                self.assertEqual(tensor[i, j], expected)
    
    def test_arange(self):
        """Test creating tensor with range."""
        tensor = tc.arange(0, 10, 2)
        expected = [0, 2, 4, 6, 8]
        self.assertEqual(tensor.data, expected)
    
    def test_linspace(self):
        """Test creating tensor with linear spacing."""
        tensor = tc.linspace(0, 1, 5)
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i, (actual, exp) in enumerate(zip(tensor.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)

class TestTensorOperations(unittest.TestCase):
    """Test tensor operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.a = tc.tensor([1, 2, 3, 4])
        self.b = tc.tensor([5, 6, 7, 8])
        self.matrix_a = tc.tensor([[1, 2], [3, 4]])
        self.matrix_b = tc.tensor([[5, 6], [7, 8]])
    
    def test_addition(self):
        """Test tensor addition."""
        result = self.a + self.b
        expected = [6, 8, 10, 12]
        self.assertEqual(result.data, expected)
    
    def test_subtraction(self):
        """Test tensor subtraction."""
        result = self.a - self.b
        expected = [-4, -4, -4, -4]
        self.assertEqual(result.data, expected)
    
    def test_multiplication(self):
        """Test tensor multiplication."""
        result = self.a * self.b
        expected = [5, 12, 21, 32]
        self.assertEqual(result.data, expected)
    
    def test_division(self):
        """Test tensor division."""
        result = self.a / self.b
        expected = [0.2, 1/3, 3/7, 0.5]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_scalar_addition(self):
        """Test scalar addition."""
        result = self.a + 10
        expected = [11, 12, 13, 14]
        self.assertEqual(result.data, expected)
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        result = self.a * 2
        expected = [2, 4, 6, 8]
        self.assertEqual(result.data, expected)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        result = self.matrix_a.matmul(self.matrix_b)
        expected = [[19, 22], [43, 50]]
        for i in range(2):
            for j in range(2):
                self.assertEqual(result[i, j], expected[i][j])
    
    def test_transpose(self):
        """Test matrix transpose."""
        result = self.matrix_a.transpose()
        expected = [[1, 3], [2, 4]]
        for i in range(2):
            for j in range(2):
                self.assertEqual(result[i, j], expected[i][j])

class TestTensorReductions(unittest.TestCase):
    """Test tensor reduction operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tensor = tc.tensor([[1, 2, 3], [4, 5, 6]])
    
    def test_sum(self):
        """Test tensor sum."""
        result = self.tensor.sum()
        self.assertEqual(result, 21)
    
    def test_sum_axis(self):
        """Test tensor sum along axis."""
        result = self.tensor.sum(0)
        expected = [5, 7, 9]
        self.assertEqual(result.data, expected)
        
        result = self.tensor.sum(1)
        expected = [6, 15]
        self.assertEqual(result.data, expected)
    
    def test_mean(self):
        """Test tensor mean."""
        result = self.tensor.mean()
        self.assertEqual(result, 3.5)
    
    def test_max(self):
        """Test tensor max."""
        result = self.tensor.max()
        self.assertEqual(result, 6)
    
    def test_min(self):
        """Test tensor min."""
        result = self.tensor.min()
        self.assertEqual(result, 1)
    
    def test_std(self):
        """Test tensor standard deviation."""
        result = self.tensor.std()
        expected_std = math.sqrt(2.9166666666666665)
        self.assertAlmostEqual(result, expected_std, places=5)

class TestTensorShapeOperations(unittest.TestCase):
    """Test tensor shape operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tensor = tc.tensor([[1, 2, 3], [4, 5, 6]])
    
    def test_reshape(self):
        """Test tensor reshape."""
        result = self.tensor.reshape((3, 2))
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.size, 6)
    
    def test_squeeze(self):
        """Test tensor squeeze."""
        tensor = tc.tensor([[[1, 2, 3]]])  # Shape: (1, 1, 3)
        result = tensor.squeeze()
        self.assertEqual(result.shape, (3,))
    
    def test_unsqueeze(self):
        """Test tensor unsqueeze."""
        tensor = tc.tensor([1, 2, 3])  # Shape: (3,)
        result = tensor.unsqueeze(0)
        self.assertEqual(result.shape, (1, 3))

class TestTensorMathematicalFunctions(unittest.TestCase):
    """Test mathematical functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tensor = tc.tensor([0, math.pi/2, math.pi, 3*math.pi/2])
    
    def test_sin(self):
        """Test sine function."""
        result = tc.sin(self.tensor)
        expected = [0, 1, 0, -1]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_cos(self):
        """Test cosine function."""
        result = tc.cos(self.tensor)
        expected = [1, 0, -1, 0]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_exp(self):
        """Test exponential function."""
        tensor = tc.tensor([0, 1, 2])
        result = tc.exp(tensor)
        expected = [1, math.e, math.e**2]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_log(self):
        """Test logarithmic function."""
        tensor = tc.tensor([1, math.e, math.e**2])
        result = tc.log(tensor)
        expected = [0, 1, 2]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_sqrt(self):
        """Test square root function."""
        tensor = tc.tensor([0, 1, 4, 9])
        result = tc.sqrt(tensor)
        expected = [0, 1, 2, 3]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)

class TestTensorBroadcasting(unittest.TestCase):
    """Test tensor broadcasting."""
    
    def test_broadcast_add(self):
        """Test broadcasting addition."""
        a = tc.tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
        b = tc.tensor([10, 20, 30])             # Shape: (3,)
        result = a + b
        expected = [[11, 22, 33], [14, 25, 36]]
        for i in range(2):
            for j in range(3):
                self.assertEqual(result[i, j], expected[i][j])
    
    def test_broadcast_multiply(self):
        """Test broadcasting multiplication."""
        a = tc.tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
        b = tc.tensor([2, 3])            # Shape: (2,)
        result = a * b
        expected = [[2, 6], [6, 12]]
        for i in range(2):
            for j in range(2):
                self.assertEqual(result[i, j], expected[i][j])

class TestTensorComparison(unittest.TestCase):
    """Test tensor comparison operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.a = tc.tensor([1, 2, 3, 4])
        self.b = tc.tensor([1, 3, 2, 4])
    
    def test_equal(self):
        """Test element-wise equality."""
        result = tc.equal(self.a, self.b)
        expected = [True, False, False, True]
        self.assertEqual(result.data, expected)
    
    def test_less(self):
        """Test element-wise less than."""
        result = tc.less(self.a, self.b)
        expected = [False, True, False, False]
        self.assertEqual(result.data, expected)
    
    def test_greater(self):
        """Test element-wise greater than."""
        result = tc.greater(self.a, self.b)
        expected = [False, False, True, False]
        self.assertEqual(result.data, expected)

class TestTensorRandom(unittest.TestCase):
    """Test random tensor generation."""
    
    def test_random_normal(self):
        """Test normal random tensor generation."""
        tensor = tc.random_normal((1000,), mean=0, std=1)
        self.assertEqual(tensor.shape, (1000,))
        
        # Check that mean is approximately 0
        mean = tensor.mean()
        self.assertAlmostEqual(mean, 0, places=1)
        
        # Check that std is approximately 1
        std = tensor.std()
        self.assertAlmostEqual(std, 1, places=1)
    
    def test_random_uniform(self):
        """Test uniform random tensor generation."""
        tensor = tc.random_uniform((1000,), min=0, max=1)
        self.assertEqual(tensor.shape, (1000,))
        
        # Check that all values are in [0, 1]
        for value in tensor.data:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)

class TestTensorMemory(unittest.TestCase):
    """Test tensor memory management."""
    
    def test_copy(self):
        """Test tensor copying."""
        original = tc.tensor([1, 2, 3, 4])
        copy = original.copy()
        
        # Modify original
        original[0] = 10
        
        # Check that copy is unchanged
        self.assertEqual(copy[0], 1)
        self.assertEqual(original[0], 10)
    
    def test_fill(self):
        """Test tensor filling."""
        tensor = tc.tensor([1, 2, 3, 4])
        tensor.fill(5)
        
        for value in tensor.data:
            self.assertEqual(value, 5)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
