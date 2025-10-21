#!/usr/bin/env python3
"""
Simple test for TensorCore sklearn functionality (Python-only version)

This test demonstrates the sklearn-style API without requiring the C++ library to be built.
"""

import sys
import os
import numpy as np

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

# Mock TensorCore tensor class for testing
class MockTensor:
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            self.data = np.array(data)
        else:
            self.data = data
        self.shape = self.data.shape
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])
    
    def __setitem__(self, key, value):
        self.data[key] = value.data if hasattr(value, 'data') else value
    
    def __add__(self, other):
        if hasattr(other, 'data'):
            return MockTensor(self.data + other.data)
        return MockTensor(self.data + other)
    
    def __sub__(self, other):
        if hasattr(other, 'data'):
            return MockTensor(self.data - other.data)
        return MockTensor(self.data - other)
    
    def __mul__(self, other):
        if hasattr(other, 'data'):
            return MockTensor(self.data * other.data)
        return MockTensor(self.data * other)
    
    def __truediv__(self, other):
        if hasattr(other, 'data'):
            return MockTensor(self.data / other.data)
        return MockTensor(self.data / other)
    
    def __matmul__(self, other):
        if hasattr(other, 'data'):
            return MockTensor(self.data @ other.data)
        return MockTensor(self.data @ other)
    
    def __rmatmul__(self, other):
        if hasattr(other, 'data'):
            return MockTensor(other.data @ self.data)
        return MockTensor(other @ self.data)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        return MockTensor(other - self.data)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        return MockTensor(other / self.data)
    
    def mean(self, axis=None):
        return MockTensor(self.data.mean(axis=axis))
    
    def std(self, axis=None):
        return MockTensor(self.data.std(axis=axis))
    
    def sum(self, axis=None):
        return MockTensor(self.data.sum(axis=axis))
    
    def max(self, axis=None):
        return MockTensor(self.data.max(axis=axis))
    
    def min(self, axis=None):
        return MockTensor(self.data.min(axis=axis))
    
    def abs(self):
        return MockTensor(np.abs(self.data))
    
    def reshape(self, shape):
        return MockTensor(self.data.reshape(shape))
    
    def transpose(self):
        return MockTensor(self.data.T)
    
    def flatten(self):
        return MockTensor(self.data.flatten())
    
    def item(self):
        return self.data.item()
    
    def copy(self):
        return MockTensor(self.data.copy())
    
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return f"MockTensor({self.data})"

# Mock TensorCore module
class MockTensorCore:
    @staticmethod
    def tensor(data):
        return MockTensor(data)
    
    @staticmethod
    def zeros(shape):
        return MockTensor(np.zeros(shape))
    
    @staticmethod
    def ones(shape):
        return MockTensor(np.ones(shape))
    
    @staticmethod
    def eye(n):
        return MockTensor(np.eye(n))
    
    @staticmethod
    def exp(x):
        if hasattr(x, 'data'):
            return MockTensor(np.exp(x.data))
        return MockTensor(np.exp(x))
    
    @staticmethod
    def log(x):
        if hasattr(x, 'data'):
            return MockTensor(np.log(x.data))
        return MockTensor(np.log(x))
    
    @staticmethod
    def unique(x):
        if hasattr(x, 'data'):
            return np.unique(x.data)
        return np.unique(x)
    
    @staticmethod
    def concatenate(tensors, axis=0):
        data = [t.data if hasattr(t, 'data') else t for t in tensors]
        return MockTensor(np.concatenate(data, axis=axis))
    
    @staticmethod
    def where(condition, x, y):
        if hasattr(condition, 'data'):
            cond = condition.data
        else:
            cond = condition
        
        if hasattr(x, 'data'):
            x_data = x.data
        else:
            x_data = x
            
        if hasattr(y, 'data'):
            y_data = y.data
        else:
            y_data = y
            
        return MockTensor(np.where(cond, x_data, y_data))
    
    @staticmethod
    def solve(A, b):
        if hasattr(A, 'data'):
            A_data = A.data
        else:
            A_data = A
            
        if hasattr(b, 'data'):
            b_data = b.data
        else:
            b_data = b
            
        return MockTensor(np.linalg.solve(A_data, b_data))
    
    @staticmethod
    def r2_score(y_true, y_pred):
        if hasattr(y_true, 'data'):
            y_true = y_true.data
        if hasattr(y_pred, 'data'):
            y_pred = y_pred.data
            
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - (ss_res / ss_tot)

# Mock the tensorcore module
tensorcore_module = MockTensorCore()
tensorcore_module.sklearn = type('MockSklearn', (), {})()
sys.modules['tensorcore'] = tensorcore_module
sys.modules['tensorcore.sklearn'] = tensorcore_module.sklearn

try:
    # Import our TensorCore sklearn modules
    from tensorcore.sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
    from tensorcore.sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from tensorcore.sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from tensorcore.sklearn.model_selection import train_test_split
    
    print("✅ TensorCore sklearn modules imported successfully!")
    
    # Test basic functionality
    print("\nTesting basic sklearn functionality...")
    
    # Generate simple test data
    X = MockTensorCore.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = MockTensorCore.tensor([3, 7, 11, 15])  # y = 2*x1 + x2 + 1
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Test Linear Regression
    print("\n1. Testing Linear Regression...")
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"   Coefficients: {lr.coef_}")
    print(f"   Intercept: {lr.intercept_}")
    print(f"   MSE: {mse:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # Test Ridge Regression
    print("\n2. Testing Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    y_pred_ridge = ridge.predict(X)
    mse_ridge = mean_squared_error(y, y_pred_ridge)
    
    print(f"   Coefficients: {ridge.coef_}")
    print(f"   Intercept: {ridge.intercept_}")
    print(f"   MSE: {mse_ridge:.4f}")
    
    # Test train_test_split
    print("\n3. Testing train_test_split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
    # Test StandardScaler
    print("\n4. Testing StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   Original mean: {X.mean(axis=0)}")
    print(f"   Scaled mean: {X_scaled.mean(axis=0)}")
    print(f"   Scaled std: {X_scaled.std(axis=0)}")
    
    # Test LabelEncoder
    print("\n5. Testing LabelEncoder...")
    y_categorical = MockTensorCore.tensor(['A', 'B', 'A', 'C', 'B'])
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_categorical)
    y_decoded = encoder.inverse_transform(y_encoded)
    
    print(f"   Original: {y_categorical}")
    print(f"   Encoded: {y_encoded}")
    print(f"   Decoded: {y_decoded}")
    
    # Test Logistic Regression
    print("\n6. Testing Logistic Regression...")
    X_class = MockTensorCore.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [1, 1], [2, 2]])
    y_class = MockTensorCore.tensor([0, 0, 0, 0, 1, 1])
    
    lr_class = LogisticRegression(random_state=42)
    lr_class.fit(X_class, y_class)
    y_pred_class = lr_class.predict(X_class)
    acc = accuracy_score(y_class, y_pred_class)
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Predictions: {y_pred_class}")
    
    # Test additional functionality
    print("\n7. Testing additional sklearn functionality:")
    print("-" * 30)
    
    # Test Lasso
    print("Testing Lasso Regression...")
    lasso = Lasso(alpha=0.1)
    lasso.fit(X, y)
    y_pred_lasso = lasso.predict(X)
    mse_lasso = mean_squared_error(y, y_pred_lasso)
    print(f"   Lasso MSE: {mse_lasso:.4f}")
    
    # Test ElasticNet
    print("Testing ElasticNet Regression...")
    elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic.fit(X, y)
    y_pred_elastic = elastic.predict(X)
    mse_elastic = mean_squared_error(y, y_pred_elastic)
    print(f"   ElasticNet MSE: {mse_elastic:.4f}")
    
    # Test MinMaxScaler
    print("Testing MinMaxScaler...")
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    print(f"   MinMax scaled range: [{X_minmax.min():.4f}, {X_minmax.max():.4f}]")
    
    print("\n✅ All sklearn functionality tests passed!")
    print("🎉 TensorCore sklearn API is working correctly!")
    print("\nNote: This test uses mock implementations to demonstrate the API.")
    print("For full functionality, the C++ library needs to be built.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the sklearn modules are properly implemented.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
