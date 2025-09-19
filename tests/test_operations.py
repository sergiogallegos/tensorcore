#!/usr/bin/env python3
"""
Test suite for TensorCore operations

This module contains tests for mathematical operations in TensorCore.
"""

import unittest
import tensorcore as tc
import math

class TestElementWiseOperations(unittest.TestCase):
    """Test element-wise operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.a = tc.tensor([1, 2, 3, 4])
        self.b = tc.tensor([5, 6, 7, 8])
    
    def test_add(self):
        """Test element-wise addition."""
        result = tc.add(self.a, self.b)
        expected = [6, 8, 10, 12]
        self.assertEqual(result.data, expected)
    
    def test_subtract(self):
        """Test element-wise subtraction."""
        result = tc.subtract(self.a, self.b)
        expected = [-4, -4, -4, -4]
        self.assertEqual(result.data, expected)
    
    def test_multiply(self):
        """Test element-wise multiplication."""
        result = tc.multiply(self.a, self.b)
        expected = [5, 12, 21, 32]
        self.assertEqual(result.data, expected)
    
    def test_divide(self):
        """Test element-wise division."""
        result = tc.divide(self.a, self.b)
        expected = [0.2, 1/3, 3/7, 0.5]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_power(self):
        """Test element-wise power."""
        result = tc.power(self.a, self.b)
        expected = [1**5, 2**6, 3**7, 4**8]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)

class TestScalarOperations(unittest.TestCase):
    """Test scalar operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tensor = tc.tensor([1, 2, 3, 4])
    
    def test_add_scalar(self):
        """Test scalar addition."""
        result = tc.add_scalar(self.tensor, 10)
        expected = [11, 12, 13, 14]
        self.assertEqual(result.data, expected)
    
    def test_subtract_scalar(self):
        """Test scalar subtraction."""
        result = tc.subtract_scalar(self.tensor, 2)
        expected = [-1, 0, 1, 2]
        self.assertEqual(result.data, expected)
    
    def test_multiply_scalar(self):
        """Test scalar multiplication."""
        result = tc.multiply_scalar(self.tensor, 3)
        expected = [3, 6, 9, 12]
        self.assertEqual(result.data, expected)
    
    def test_divide_scalar(self):
        """Test scalar division."""
        result = tc.divide_scalar(self.tensor, 2)
        expected = [0.5, 1, 1.5, 2]
        self.assertEqual(result.data, expected)
    
    def test_power_scalar(self):
        """Test scalar power."""
        result = tc.power_scalar(self.tensor, 2)
        expected = [1, 4, 9, 16]
        self.assertEqual(result.data, expected)

class TestTrigonometricFunctions(unittest.TestCase):
    """Test trigonometric functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.angles = tc.tensor([0, math.pi/6, math.pi/4, math.pi/3, math.pi/2])
    
    def test_sin(self):
        """Test sine function."""
        result = tc.sin(self.angles)
        expected = [0, 0.5, math.sqrt(2)/2, math.sqrt(3)/2, 1]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_cos(self):
        """Test cosine function."""
        result = tc.cos(self.angles)
        expected = [1, math.sqrt(3)/2, math.sqrt(2)/2, 0.5, 0]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_tan(self):
        """Test tangent function."""
        result = tc.tan(self.angles)
        expected = [0, 1/math.sqrt(3), 1, math.sqrt(3), float('inf')]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            if exp == float('inf'):
                self.assertTrue(math.isinf(actual))
            else:
                self.assertAlmostEqual(actual, exp, places=5)
    
    def test_asin(self):
        """Test arcsine function."""
        values = tc.tensor([0, 0.5, math.sqrt(2)/2, math.sqrt(3)/2, 1])
        result = tc.asin(values)
        expected = [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_acos(self):
        """Test arccosine function."""
        values = tc.tensor([1, math.sqrt(3)/2, math.sqrt(2)/2, 0.5, 0])
        result = tc.acos(values)
        expected = [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_atan(self):
        """Test arctangent function."""
        values = tc.tensor([0, 1/math.sqrt(3), 1, math.sqrt(3)])
        result = tc.atan(values)
        expected = [0, math.pi/6, math.pi/4, math.pi/3]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)

class TestHyperbolicFunctions(unittest.TestCase):
    """Test hyperbolic functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.values = tc.tensor([0, 1, 2])
    
    def test_sinh(self):
        """Test hyperbolic sine function."""
        result = tc.sinh(self.values)
        expected = [0, math.sinh(1), math.sinh(2)]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_cosh(self):
        """Test hyperbolic cosine function."""
        result = tc.cosh(self.values)
        expected = [1, math.cosh(1), math.cosh(2)]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_tanh(self):
        """Test hyperbolic tangent function."""
        result = tc.tanh(self.values)
        expected = [0, math.tanh(1), math.tanh(2)]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)

class TestLogarithmicFunctions(unittest.TestCase):
    """Test logarithmic functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.values = tc.tensor([1, math.e, math.e**2, 10])
    
    def test_log(self):
        """Test natural logarithm."""
        result = tc.log(self.values)
        expected = [0, 1, 2, math.log(10)]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_log2(self):
        """Test base-2 logarithm."""
        result = tc.log2(self.values)
        expected = [0, 1/math.log(2), 2/math.log(2), math.log(10)/math.log(2)]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_log10(self):
        """Test base-10 logarithm."""
        result = tc.log10(self.values)
        expected = [0, 1/math.log(10), 2/math.log(10), 1]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_exp(self):
        """Test exponential function."""
        values = tc.tensor([0, 1, 2])
        result = tc.exp(values)
        expected = [1, math.e, math.e**2]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)

class TestPowerFunctions(unittest.TestCase):
    """Test power functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.values = tc.tensor([0, 1, 4, 9, 16])
    
    def test_sqrt(self):
        """Test square root function."""
        result = tc.sqrt(self.values)
        expected = [0, 1, 2, 3, 4]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_cbrt(self):
        """Test cube root function."""
        values = tc.tensor([0, 1, 8, 27])
        result = tc.cbrt(values)
        expected = [0, 1, 2, 3]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_square(self):
        """Test square function."""
        values = tc.tensor([0, 1, 2, 3, 4])
        result = tc.square(values)
        expected = [0, 1, 4, 9, 16]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_reciprocal(self):
        """Test reciprocal function."""
        values = tc.tensor([1, 2, 4, 8])
        result = tc.reciprocal(values)
        expected = [1, 0.5, 0.25, 0.125]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)

class TestRoundingFunctions(unittest.TestCase):
    """Test rounding functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.values = tc.tensor([-2.7, -1.5, -0.5, 0.5, 1.5, 2.7])
    
    def test_floor(self):
        """Test floor function."""
        result = tc.floor(self.values)
        expected = [-3, -2, -1, 0, 1, 2]
        self.assertEqual(result.data, expected)
    
    def test_ceil(self):
        """Test ceiling function."""
        result = tc.ceil(self.values)
        expected = [-2, -1, 0, 1, 2, 3]
        self.assertEqual(result.data, expected)
    
    def test_round(self):
        """Test round function."""
        result = tc.round(self.values)
        expected = [-3, -2, 0, 0, 2, 3]
        self.assertEqual(result.data, expected)
    
    def test_trunc(self):
        """Test truncate function."""
        result = tc.trunc(self.values)
        expected = [-2, -1, 0, 0, 1, 2]
        self.assertEqual(result.data, expected)

class TestComparisonOperations(unittest.TestCase):
    """Test comparison operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.a = tc.tensor([1, 2, 3, 4])
        self.b = tc.tensor([1, 3, 2, 4])
    
    def test_equal(self):
        """Test element-wise equality."""
        result = tc.equal(self.a, self.b)
        expected = [True, False, False, True]
        self.assertEqual(result.data, expected)
    
    def test_not_equal(self):
        """Test element-wise inequality."""
        result = tc.not_equal(self.a, self.b)
        expected = [False, True, True, False]
        self.assertEqual(result.data, expected)
    
    def test_less(self):
        """Test element-wise less than."""
        result = tc.less(self.a, self.b)
        expected = [False, True, False, False]
        self.assertEqual(result.data, expected)
    
    def test_less_equal(self):
        """Test element-wise less than or equal."""
        result = tc.less_equal(self.a, self.b)
        expected = [True, True, False, True]
        self.assertEqual(result.data, expected)
    
    def test_greater(self):
        """Test element-wise greater than."""
        result = tc.greater(self.a, self.b)
        expected = [False, False, True, False]
        self.assertEqual(result.data, expected)
    
    def test_greater_equal(self):
        """Test element-wise greater than or equal."""
        result = tc.greater_equal(self.a, self.b)
        expected = [True, False, True, True]
        self.assertEqual(result.data, expected)

class TestLogicalOperations(unittest.TestCase):
    """Test logical operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.a = tc.tensor([True, False, True, False])
        self.b = tc.tensor([True, True, False, False])
    
    def test_logical_and(self):
        """Test logical AND."""
        result = tc.logical_and(self.a, self.b)
        expected = [True, False, False, False]
        self.assertEqual(result.data, expected)
    
    def test_logical_or(self):
        """Test logical OR."""
        result = tc.logical_or(self.a, self.b)
        expected = [True, True, True, False]
        self.assertEqual(result.data, expected)
    
    def test_logical_xor(self):
        """Test logical XOR."""
        result = tc.logical_xor(self.a, self.b)
        expected = [False, True, True, False]
        self.assertEqual(result.data, expected)
    
    def test_logical_not(self):
        """Test logical NOT."""
        result = tc.logical_not(self.a)
        expected = [False, True, False, True]
        self.assertEqual(result.data, expected)

class TestLinearAlgebraOperations(unittest.TestCase):
    """Test linear algebra operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.A = tc.tensor([[1, 2], [3, 4]])
        self.B = tc.tensor([[5, 6], [7, 8]])
        self.vector = tc.tensor([1, 2, 3])
    
    def test_matmul(self):
        """Test matrix multiplication."""
        result = tc.matmul(self.A, self.B)
        expected = [[19, 22], [43, 50]]
        for i in range(2):
            for j in range(2):
                self.assertEqual(result[i, j], expected[i][j])
    
    def test_dot(self):
        """Test dot product."""
        a = tc.tensor([1, 2, 3])
        b = tc.tensor([4, 5, 6])
        result = tc.dot(a, b)
        expected = 1*4 + 2*5 + 3*6  # 32
        self.assertEqual(result, expected)
    
    def test_outer(self):
        """Test outer product."""
        a = tc.tensor([1, 2])
        b = tc.tensor([3, 4])
        result = tc.outer(a, b)
        expected = [[3, 4], [6, 8]]
        for i in range(2):
            for j in range(2):
                self.assertEqual(result[i, j], expected[i][j])
    
    def test_cross(self):
        """Test cross product."""
        a = tc.tensor([1, 0, 0])
        b = tc.tensor([0, 1, 0])
        result = tc.cross(a, b)
        expected = [0, 0, 1]
        for i, (actual, exp) in enumerate(zip(result.data, expected)):
            self.assertAlmostEqual(actual, exp, places=5)
    
    def test_norm(self):
        """Test vector norm."""
        result = tc.norm(self.vector)
        expected = math.sqrt(1**2 + 2**2 + 3**2)
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_transpose(self):
        """Test matrix transpose."""
        result = tc.transpose(self.A)
        expected = [[1, 3], [2, 4]]
        for i in range(2):
            for j in range(2):
                self.assertEqual(result[i, j], expected[i][j])
    
    def test_trace(self):
        """Test matrix trace."""
        result = tc.trace(self.A)
        expected = 1 + 4  # 5
        self.assertEqual(result, expected)
    
    def test_det(self):
        """Test matrix determinant."""
        result = tc.det(self.A)
        expected = 1*4 - 2*3  # -2
        self.assertEqual(result, expected)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
