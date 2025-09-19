#!/usr/bin/env python3
"""
Linear Regression Example

This example demonstrates how to implement linear regression from scratch
using TensorCore, showing the mathematical foundations behind the algorithm.
"""

import tensorcore as tc
import time
import matplotlib.pyplot as plt
import numpy as np

def generate_data(n_samples=100, noise=0.1):
    """Generate synthetic linear regression data."""
    # True parameters
    true_slope = 2.5
    true_intercept = 1.0
    
    # Generate features
    X = tc.random_uniform((n_samples, 1), min=-2, max=2)
    
    # Generate targets with noise
    y = true_slope * X[:, 0] + true_intercept + noise * tc.random_normal((n_samples,))
    
    return X, y, true_slope, true_intercept

def mean_squared_error(y_true, y_pred):
    """Compute mean squared error."""
    return ((y_true - y_pred) ** 2).mean()

def mean_squared_error_gradient(y_true, y_pred):
    """Compute gradient of MSE."""
    return 2 * (y_pred - y_true) / len(y_true)

def linear_regression_analytical(X, y):
    """Solve linear regression using analytical solution."""
    print("Analytical Solution (Normal Equation)")
    print("-" * 35)
    
    # Add bias term (intercept)
    X_with_bias = tc.concatenate([tc.ones((X.shape[0], 1)), X], axis=1)
    
    # Normal equation: θ = (X^T X)^(-1) X^T y
    XTX = X_with_bias.transpose().matmul(X_with_bias)
    XTy = X_with_bias.transpose().matmul(y.reshape(-1, 1))
    
    # Solve for parameters
    theta = XTX.inv().matmul(XTy)
    
    intercept = theta[0, 0]
    slope = theta[1, 0]
    
    print(f"Intercept: {intercept:.4f}")
    print(f"Slope: {slope:.4f}")
    
    return intercept, slope

def linear_regression_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """Solve linear regression using gradient descent."""
    print(f"\nGradient Descent Solution")
    print("-" * 25)
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")
    
    # Initialize parameters
    intercept = tc.tensor(0.0)
    slope = tc.tensor(0.0)
    
    # Training history
    losses = []
    
    start_time = time.time()
    
    for iteration in range(n_iterations):
        # Forward pass: y_pred = slope * X + intercept
        y_pred = slope * X[:, 0] + intercept
        
        # Compute loss
        loss = mean_squared_error(y, y_pred)
        losses.append(loss.item())
        
        # Compute gradients
        grad_intercept = mean_squared_error_gradient(y, y_pred).sum()
        grad_slope = (mean_squared_error_gradient(y, y_pred) * X[:, 0]).sum()
        
        # Update parameters
        intercept = intercept - learning_rate * grad_intercept
        slope = slope - learning_rate * grad_slope
        
        # Print progress
        if iteration % 200 == 0:
            print(f"Iteration {iteration:4d}: Loss = {loss:.6f}, "
                  f"Intercept = {intercept:.4f}, Slope = {slope:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.4f} seconds")
    print(f"Final - Intercept: {intercept:.4f}, Slope: {slope:.4f}")
    
    return intercept, slope, losses

def linear_regression_stochastic_gd(X, y, learning_rate=0.01, n_epochs=100, batch_size=10):
    """Solve linear regression using stochastic gradient descent."""
    print(f"\nStochastic Gradient Descent")
    print("-" * 30)
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    
    # Initialize parameters
    intercept = tc.tensor(0.0)
    slope = tc.tensor(0.0)
    
    # Training history
    losses = []
    n_samples = X.shape[0]
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle data
        indices = tc.tensor(list(range(n_samples)))
        indices = tc.tensor(np.random.permutation(indices.data))
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Forward pass
            y_pred = slope * batch_X[:, 0] + intercept
            
            # Compute loss
            loss = mean_squared_error(batch_y, y_pred)
            epoch_loss += loss.item()
            n_batches += 1
            
            # Compute gradients
            grad_intercept = mean_squared_error_gradient(batch_y, y_pred).sum()
            grad_slope = (mean_squared_error_gradient(batch_y, y_pred) * batch_X[:, 0]).sum()
            
            # Update parameters
            intercept = intercept - learning_rate * grad_intercept
            slope = slope - learning_rate * grad_slope
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        # Print progress
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}, "
                  f"Intercept = {intercept:.4f}, Slope = {slope:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.4f} seconds")
    print(f"Final - Intercept: {intercept:.4f}, Slope: {slope:.4f}")
    
    return intercept, slope, losses

def plot_results(X, y, true_slope, true_intercept, 
                analytical_params, gd_params, sgd_params, gd_losses, sgd_losses):
    """Plot the results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Convert to numpy for plotting
        X_np = X.data
        y_np = y.data
        
        # Plot 1: Data and fitted lines
        ax1.scatter(X_np, y_np, alpha=0.6, label='Data')
        
        x_range = np.linspace(X_np.min(), X_np.max(), 100)
        
        # True line
        y_true = true_slope * x_range + true_intercept
        ax1.plot(x_range, y_true, 'r-', label=f'True (slope={true_slope:.2f})', linewidth=2)
        
        # Analytical solution
        y_analytical = analytical_params[1] * x_range + analytical_params[0]
        ax1.plot(x_range, y_analytical, 'g--', label=f'Analytical (slope={analytical_params[1]:.2f})', linewidth=2)
        
        # Gradient descent
        y_gd = gd_params[1] * x_range + gd_params[0]
        ax1.plot(x_range, y_gd, 'b:', label=f'GD (slope={gd_params[1]:.2f})', linewidth=2)
        
        # Stochastic gradient descent
        y_sgd = sgd_params[1] * x_range + sgd_params[0]
        ax1.plot(x_range, y_sgd, 'm-.', label=f'SGD (slope={sgd_params[1]:.2f})', linewidth=2)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.set_title('Linear Regression Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss curves
        ax2.plot(gd_losses, label='Gradient Descent', color='blue')
        ax2.plot(sgd_losses, label='Stochastic GD', color='magenta')
        ax2.set_xlabel('Iteration/Epoch')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Residuals (Analytical)
        y_pred_analytical = analytical_params[1] * X_np + analytical_params[0]
        residuals_analytical = y_np - y_pred_analytical
        ax3.scatter(y_pred_analytical, residuals_analytical, alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals (Analytical)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residuals (Gradient Descent)
        y_pred_gd = gd_params[1] * X_np + gd_params[0]
        residuals_gd = y_np - y_pred_gd
        ax4.scatter(y_pred_gd, residuals_gd, alpha=0.6)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals (Gradient Descent)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping plots.")

def compare_with_sklearn(X, y):
    """Compare with scikit-learn implementation."""
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error as sklearn_mse
        
        print("\nComparison with Scikit-learn")
        print("-" * 30)
        
        # Convert to numpy
        X_np = X.data
        y_np = y.data
        
        # Fit with sklearn
        model = LinearRegression()
        model.fit(X_np, y_np)
        
        # Predictions
        y_pred = model.predict(X_np)
        mse = sklearn_mse(y_np, y_pred)
        
        print(f"Sklearn - Intercept: {model.intercept_:.4f}")
        print(f"Sklearn - Slope: {model.coef_[0]:.4f}")
        print(f"Sklearn - MSE: {mse:.6f}")
        
    except ImportError:
        print("Scikit-learn not available. Skipping comparison.")

def main():
    print("TensorCore Linear Regression Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    tc.set_random_seed(42)
    
    # Generate data
    X, y, true_slope, true_intercept = generate_data(n_samples=200, noise=0.1)
    print(f"Generated {X.shape[0]} samples")
    print(f"True parameters - Intercept: {true_intercept:.4f}, Slope: {true_slope:.4f}")
    
    # Analytical solution
    analytical_params = linear_regression_analytical(X, y)
    
    # Gradient descent
    gd_params = linear_regression_gradient_descent(X, y, learning_rate=0.01, n_iterations=1000)
    
    # Stochastic gradient descent
    sgd_params = linear_regression_stochastic_gd(X, y, learning_rate=0.01, n_epochs=100, batch_size=20)
    
    # Compare with sklearn
    compare_with_sklearn(X, y)
    
    # Plot results
    plot_results(X, y, true_slope, true_intercept, 
                analytical_params, gd_params, sgd_params, 
                gd_params[2], sgd_params[2])
    
    print("\n" + "=" * 40)
    print("Linear regression example completed! 📈")

if __name__ == "__main__":
    main()
