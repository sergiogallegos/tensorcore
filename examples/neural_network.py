#!/usr/bin/env python3
"""
Neural Network Example

This example demonstrates how to build and train a simple neural network
using TensorCore for educational purposes.
"""

import tensorcore as tc
from tensorcore.nn import Dense, ReLU, Sigmoid, Sequential
from tensorcore.nn import MSELoss, CrossEntropyLoss
from tensorcore.optim import SGD, Adam
import time

def create_synthetic_data(n_samples=1000, n_features=2, n_classes=2):
    """Create synthetic classification data."""
    # Generate random features
    X = tc.random_normal((n_samples, n_features))
    
    # Create simple decision boundary: x1 + x2 > 0
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X, y

def create_regression_data(n_samples=1000, n_features=1):
    """Create synthetic regression data."""
    # Generate random features
    X = tc.random_normal((n_samples, n_features))
    
    # Create linear relationship with noise
    y = 2 * X[:, 0] + 1 + 0.1 * tc.random_normal((n_samples,))
    
    return X, y

def train_classifier():
    """Train a binary classifier."""
    print("Binary Classification Example")
    print("=" * 30)
    
    # Create data
    X, y = create_synthetic_data(n_samples=1000, n_features=2)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create model
    model = Sequential([
        Dense(2, 10, activation=ReLU),
        Dense(10, 5, activation=ReLU),
        Dense(5, 1, activation=Sigmoid)
    ])
    
    # Loss function and optimizer
    criterion = MSELoss()
    optimizer = Adam(model.parameters, lr=0.01)
    
    # Training loop
    n_epochs = 100
    batch_size = 32
    
    print(f"\nTraining for {n_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        # Simple batch training (in practice, you'd use a proper DataLoader)
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size].float()
            
            # Forward pass
            predictions = model.forward(batch_X)
            loss = criterion.forward(predictions, batch_y)
            
            # Backward pass
            grad_output = criterion.backward(predictions, batch_y)
            model.backward(grad_output)
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Test the model
    test_X, test_y = create_synthetic_data(n_samples=200, n_features=2)
    with model.eval():
        predictions = model.forward(test_X)
        predictions = (predictions > 0.5).astype(int)
        accuracy = (predictions == test_y).float().mean()
        print(f"Test accuracy: {accuracy:.4f}")

def train_regressor():
    """Train a linear regressor."""
    print("\nLinear Regression Example")
    print("=" * 25)
    
    # Create data
    X, y = create_regression_data(n_samples=1000, n_features=1)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create model
    model = Sequential([
        Dense(1, 10, activation=ReLU),
        Dense(10, 1)
    ])
    
    # Loss function and optimizer
    criterion = MSELoss()
    optimizer = SGD(model.parameters, lr=0.01)
    
    # Training loop
    n_epochs = 50
    batch_size = 32
    
    print(f"\nTraining for {n_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Forward pass
            predictions = model.forward(batch_X)
            loss = criterion.forward(predictions, batch_y)
            
            # Backward pass
            grad_output = criterion.backward(predictions, batch_y)
            model.backward(grad_output)
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Test the model
    test_X, test_y = create_regression_data(n_samples=200, n_features=1)
    with model.eval():
        predictions = model.forward(test_X)
        mse = ((predictions - test_y) ** 2).mean()
        print(f"Test MSE: {mse:.4f}")

def demonstrate_activations():
    """Demonstrate different activation functions."""
    print("\nActivation Functions Demo")
    print("=" * 25)
    
    from tensorcore.nn import ReLU, Sigmoid, Tanh, Softmax
    
    # Create test data
    x = tc.tensor([-2, -1, 0, 1, 2])
    print(f"Input: {x}")
    
    # Test different activations
    activations = [
        ("ReLU", ReLU),
        ("Sigmoid", Sigmoid),
        ("Tanh", Tanh),
    ]
    
    for name, activation in activations:
        output = activation.forward(x)
        print(f"{name:8s}: {output}")
    
    # Test softmax
    x_2d = tc.tensor([[1, 2, 3], [4, 5, 6]])
    softmax_output = Softmax.forward(x_2d)
    print(f"Softmax input:\n{x_2d}")
    print(f"Softmax output:\n{softmax_output}")

def demonstrate_optimizers():
    """Demonstrate different optimizers."""
    print("\nOptimizers Demo")
    print("=" * 15)
    
    from tensorcore.optim import SGD, Adam, RMSprop
    
    # Create a simple loss function
    def simple_loss(params):
        return (params[0] ** 2 + params[1] ** 2).sum()
    
    # Test different optimizers
    optimizers = [
        ("SGD", SGD),
        ("Adam", Adam),
        ("RMSprop", RMSprop),
    ]
    
    for name, OptimizerClass in optimizers:
        print(f"\n{name} optimization:")
        
        # Initialize parameters
        params = [tc.tensor.random_normal((2, 2)) for _ in range(2)]
        optimizer = OptimizerClass(params, lr=0.1)
        
        # Optimize for a few steps
        for step in range(10):
            loss = simple_loss(params)
            if step % 5 == 0:
                print(f"  Step {step:2d}: Loss = {loss:.4f}")
            
            # Compute gradients (simplified)
            for param in params:
                param.grad = 2 * param
            
            optimizer.step()
            optimizer.zero_grad()

def main():
    print("TensorCore Neural Network Example")
    print("=" * 35)
    
    # Set random seed for reproducibility
    tc.set_random_seed(42)
    
    # Run examples
    train_classifier()
    train_regressor()
    demonstrate_activations()
    demonstrate_optimizers()
    
    print("\n" + "=" * 35)
    print("Neural network example completed! 🧠")

if __name__ == "__main__":
    main()
