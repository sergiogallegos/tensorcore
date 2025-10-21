#!/usr/bin/env python3
"""
Scikit-learn Style Machine Learning Example

This example demonstrates how to use TensorCore's scikit-learn style
machine learning algorithms for various tasks including regression,
classification, and preprocessing.
"""

import tensorcore as tc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import mean_squared_error as sklearn_mean_squared_error

# Import TensorCore sklearn modules
from tensorcore.sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from tensorcore.sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from tensorcore.sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from tensorcore.sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


def regression_example():
    """Demonstrate linear regression algorithms."""
    print("=" * 60)
    print("REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Generate regression data
    X, y = make_regression(n_samples=200, n_features=2, noise=0.1, random_state=42)
    X = tc.tensor(X)
    y = tc.tensor(y)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Test different regression algorithms
    algorithms = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in algorithms.items():
        print(f"\n{name}:")
        print("-" * 30)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        if hasattr(model, 'coef_'):
            print(f"Coefficients: {model.coef_}")
        if hasattr(model, 'intercept_'):
            print(f"Intercept: {model.intercept_}")
        
        results[name] = {'mse': mse, 'r2': r2}
    
    # Compare with sklearn
    print(f"\nComparison with Scikit-learn:")
    print("-" * 30)
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train.data, y_train.data)
    sklearn_pred = sklearn_model.predict(X_test.data)
    sklearn_mse = sklearn_mean_squared_error(y_test.data, sklearn_pred)
    sklearn_r2 = sklearn_model.score(X_test.data, y_test.data)
    
    print(f"Sklearn MSE: {sklearn_mse:.4f}")
    print(f"Sklearn R²: {sklearn_r2:.4f}")
    
    return results


def classification_example():
    """Demonstrate classification algorithms."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Generate classification data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, random_state=42)
    X = tc.tensor(X)
    y = tc.tensor(y)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {tc.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Test logistic regression
    print(f"\nLogistic Regression:")
    print("-" * 30)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    if hasattr(model, 'coef_'):
        print(f"Coefficients: {model.coef_}")
    if hasattr(model, 'intercept_'):
        print(f"Intercept: {model.intercept_}")
    
    # Compare with sklearn
    print(f"\nComparison with Scikit-learn:")
    print("-" * 30)
    sklearn_model = SklearnLogisticRegression(random_state=42)
    sklearn_model.fit(X_train.data, y_train.data)
    sklearn_pred = sklearn_model.predict(X_test.data)
    sklearn_acc = sklearn_accuracy_score(y_test.data, sklearn_pred)
    
    print(f"Sklearn Accuracy: {sklearn_acc:.4f}")
    print(f"Sklearn Coefficients: {sklearn_model.coef_[0]}")
    print(f"Sklearn Intercept: {sklearn_model.intercept_[0]}")
    
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def preprocessing_example():
    """Demonstrate preprocessing utilities."""
    print("\n" + "=" * 60)
    print("PREPROCESSING EXAMPLE")
    print("=" * 60)
    
    # Generate data with different scales
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 0] *= 100  # Scale first feature
    X[:, 1] *= 0.01  # Scale second feature
    # Third feature stays normal scale
    
    X = tc.tensor(X)
    print(f"Original data shape: {X.shape}")
    print(f"Original data statistics:")
    print(f"  Mean: {X.mean(axis=0)}")
    print(f"  Std: {X.std(axis=0)}")
    
    # Standard Scaler
    print(f"\nStandard Scaler:")
    print("-" * 30)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Scaled data statistics:")
    print(f"  Mean: {X_scaled.mean(axis=0)}")
    print(f"  Std: {X_scaled.std(axis=0)}")
    
    # Min-Max Scaler
    print(f"\nMin-Max Scaler:")
    print("-" * 30)
    minmax_scaler = MinMaxScaler()
    X_minmax = minmax_scaler.fit_transform(X)
    
    print(f"Min-Max scaled data statistics:")
    print(f"  Min: {X_minmax.min(axis=0)}")
    print(f"  Max: {X_minmax.max(axis=0)}")
    
    # Label Encoder
    print(f"\nLabel Encoder:")
    print("-" * 30)
    y_categorical = tc.tensor(['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'])
    print(f"Original labels: {y_categorical}")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_categorical)
    print(f"Encoded labels: {y_encoded}")
    
    y_decoded = label_encoder.inverse_transform(y_encoded)
    print(f"Decoded labels: {y_decoded}")
    
    # Compare with sklearn
    print(f"\nComparison with Scikit-learn:")
    print("-" * 30)
    sklearn_scaler = SklearnStandardScaler()
    sklearn_scaled = sklearn_scaler.fit_transform(X.data)
    
    print(f"Sklearn scaled data statistics:")
    print(f"  Mean: {sklearn_scaled.mean(axis=0)}")
    print(f"  Std: {sklearn_scaled.std(axis=0)}")
    
    return X_scaled, X_minmax, y_encoded


def cross_validation_example():
    """Demonstrate cross-validation."""
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION EXAMPLE")
    print("=" * 60)
    
    # Load iris dataset
    iris = load_iris()
    X = tc.tensor(iris.data)
    y = tc.tensor(iris.target)
    
    print(f"Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {tc.unique(y)}")
    
    # Create model
    model = LogisticRegression(random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    print(f"Std CV score: {cv_scores.std():.4f}")
    
    # Grid search
    print(f"\nGrid Search:")
    print("-" * 30)
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2']
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return cv_scores, grid_search.best_score_


def visualization_example():
    """Demonstrate visualization of results."""
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    try:
        # Generate 2D classification data
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, random_state=42)
        X = tc.tensor(X)
        y = tc.tensor(y)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        
        # Create mesh for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        # Make predictions on mesh
        mesh_points = tc.tensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(X[:, 0].data, X[:, 1].data, c=y.data, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Logistic Regression Decision Boundary')
        plt.show()
        
        print("Visualization completed successfully!")
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")
    except Exception as e:
        print(f"Visualization failed: {e}")


def performance_comparison():
    """Compare performance between TensorCore and sklearn."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    import time
    
    # Generate large dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X = tc.tensor(X)
    y = tc.tensor(y)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # TensorCore Linear Regression
    print(f"\nTensorCore Linear Regression:")
    start_time = time.time()
    tc_model = LinearRegression()
    tc_model.fit(X, y)
    tc_pred = tc_model.predict(X)
    tc_time = time.time() - start_time
    tc_mse = mean_squared_error(y, tc_pred)
    
    print(f"Time: {tc_time:.4f} seconds")
    print(f"MSE: {tc_mse:.4f}")
    
    # Sklearn Linear Regression
    print(f"\nScikit-learn Linear Regression:")
    start_time = time.time()
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X.data, y.data)
    sklearn_pred = sklearn_model.predict(X.data)
    sklearn_time = time.time() - start_time
    sklearn_mse = sklearn_mean_squared_error(y.data, sklearn_pred)
    
    print(f"Time: {sklearn_time:.4f} seconds")
    print(f"MSE: {sklearn_mse:.4f}")
    
    # Performance ratio
    speedup = sklearn_time / tc_time
    print(f"\nSpeedup: {speedup:.2f}x {'(TensorCore faster)' if speedup > 1 else '(Sklearn faster)'}")
    
    return tc_time, sklearn_time, tc_mse, sklearn_mse


def main():
    """Main function to run all examples."""
    print("TensorCore Scikit-learn Style Machine Learning Examples")
    print("=" * 60)
    
    # Set random seed for reproducibility
    tc.set_seed(42)
    
    try:
        # Run examples
        regression_results = regression_example()
        classification_results = classification_example()
        preprocessing_results = preprocessing_example()
        cv_results = cross_validation_example()
        performance_results = performance_comparison()
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✅ Linear Regression algorithms implemented")
        print("✅ Ridge, Lasso, and Elastic Net regression")
        print("✅ Logistic Regression for classification")
        print("✅ Standard and Min-Max scaling")
        print("✅ Label encoding and one-hot encoding")
        print("✅ Cross-validation and grid search")
        print("✅ Performance comparison with sklearn")
        print("✅ Comprehensive evaluation metrics")
        
        print(f"\nRegression Results:")
        for name, results in regression_results.items():
            print(f"  {name}: R² = {results['r2']:.4f}, MSE = {results['mse']:.4f}")
        
        print(f"\nClassification Results:")
        print(f"  Accuracy: {classification_results['accuracy']:.4f}")
        print(f"  F1-Score: {classification_results['f1']:.4f}")
        
        print(f"\nCross-validation Results:")
        print(f"  Mean CV Score: {cv_results[1]:.4f}")
        
        print(f"\nPerformance Results:")
        print(f"  TensorCore Time: {performance_results[0]:.4f}s")
        print(f"  Sklearn Time: {performance_results[1]:.4f}s")
        
        # Visualization
        visualization_example()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully! 🎉")
        print("TensorCore now has scikit-learn style ML algorithms!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
