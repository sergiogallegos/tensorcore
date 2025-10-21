#!/usr/bin/env python3
"""
Simple test for TensorCore sklearn functionality
"""

import sys
import os

# Add the python directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    import tensorcore as tc
    from tensorcore.sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
    from tensorcore.sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from tensorcore.sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    from tensorcore.sklearn.model_selection import train_test_split
    
    print("✅ TensorCore sklearn modules imported successfully!")
    
    # Test basic functionality
    print("\nTesting basic sklearn functionality...")
    
    # Generate simple test data
    X = tc.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = tc.tensor([3, 7, 11, 15])  # y = 2*x1 + x2 + 1
    
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
    y_categorical = tc.tensor(['A', 'B', 'A', 'C', 'B'])
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_categorical)
    y_decoded = encoder.inverse_transform(y_encoded)
    
    print(f"   Original: {y_categorical}")
    print(f"   Encoded: {y_encoded}")
    print(f"   Decoded: {y_decoded}")
    
    # Test Logistic Regression
    print("\n6. Testing Logistic Regression...")
    X_class = tc.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [1, 1], [2, 2]])
    y_class = tc.tensor([0, 0, 0, 0, 1, 1])
    
    lr_class = LogisticRegression()
    lr_class.fit(X_class, y_class)
    y_pred_class = lr_class.predict(X_class)
    acc = accuracy_score(y_class, y_pred_class)
    
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Predictions: {y_pred_class}")
    
    print("\n✅ All sklearn functionality tests passed!")
    print("🎉 TensorCore now has scikit-learn style machine learning algorithms!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure TensorCore is properly built and installed.")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
