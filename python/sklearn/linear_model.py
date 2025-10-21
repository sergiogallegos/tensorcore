"""
Linear models for TensorCore

This module provides linear regression and classification models
following the scikit-learn API.
"""

import tensorcore as tc
from typing import Optional, Union, Tuple
import numpy as np


class BaseLinearModel:
    """Base class for linear models."""
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.fitted_ = False
    
    def fit(self, X, y):
        """Fit the model to training data."""
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        return self._predict(X)
    
    def _predict(self, X):
        """Internal prediction method."""
        raise NotImplementedError
    
    def score(self, X, y):
        """Return the coefficient of determination R^2."""
        y_pred = self.predict(X)
        return tc.r2_score(y, y_pred)


class LinearRegression(BaseLinearModel):
    """
    Ordinary least squares linear regression.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    normalize : bool, default=False
        Whether to normalize the regressors.
    copy_X : bool, default=True
        Whether to copy X.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    """
    
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
    
    def fit(self, X, y):
        """Fit linear model."""
        # Convert to TensorCore tensors if needed
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        n_samples, n_features = X.shape
        
        # Normalize features if requested
        if self.normalize:
            self.X_mean_ = X.mean(axis=0)
            self.X_std_ = X.std(axis=0)
            X = (X - self.X_mean_) / self.X_std_
        
        if self.fit_intercept:
            # Add bias term
            ones = tc.ones((n_samples, 1))
            X = tc.concatenate([ones, X], axis=1)
            n_features += 1
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        XTX = X.T @ X
        XTy = X.T @ y.reshape(-1, 1)
        
        # Solve for parameters
        theta = tc.solve(XTX, XTy)
        
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = tc.zeros(1)
            self.coef_ = theta
        
        self.fitted_ = True
        return self
    
    def _predict(self, X):
        """Make predictions."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        # Normalize features if they were normalized during training
        if self.normalize:
            X = (X - self.X_mean_) / self.X_std_
        
        predictions = X @ self.coef_
        if self.fit_intercept:
            predictions = predictions + self.intercept_
        
        return predictions


class Ridge(BaseLinearModel):
    """
    Ridge regression with L2 regularization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    normalize : bool, default=False
        Whether to normalize the regressors.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    """
    
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, 
                 max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        """Fit ridge regression model."""
        # Convert to TensorCore tensors if needed
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        n_samples, n_features = X.shape
        
        # Normalize features if requested
        if self.normalize:
            self.X_mean_ = X.mean(axis=0)
            self.X_std_ = X.std(axis=0)
            X = (X - self.X_mean_) / self.X_std_
        
        if self.fit_intercept:
            # Add bias term
            ones = tc.ones((n_samples, 1))
            X = tc.concatenate([ones, X], axis=1)
            n_features += 1
        
        # Ridge regression: θ = (X^T X + αI)^(-1) X^T y
        XTX = X.T @ X
        I = tc.eye(n_features) * self.alpha
        XTy = X.T @ y.reshape(-1, 1)
        
        # Solve for parameters
        theta = tc.solve(XTX + I, XTy)
        
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = tc.zeros(1)
            self.coef_ = theta
        
        self.fitted_ = True
        return self
    
    def _predict(self, X):
        """Make predictions."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        # Normalize features if they were normalized during training
        if self.normalize:
            X = (X - self.X_mean_) / self.X_std_
        
        predictions = X @ self.coef_
        if self.fit_intercept:
            predictions = predictions + self.intercept_
        
        return predictions


class Lasso(BaseLinearModel):
    """
    Lasso regression with L1 regularization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    normalize : bool, default=False
        Whether to normalize the regressors.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    """
    
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, 
                 max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        """Fit lasso regression model using coordinate descent."""
        # Convert to TensorCore tensors if needed
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        n_samples, n_features = X.shape
        
        # Normalize features if requested
        if self.normalize:
            self.X_mean_ = X.mean(axis=0)
            self.X_std_ = X.std(axis=0)
            X = (X - self.X_mean_) / self.X_std_
        
        # Initialize coefficients
        self.coef_ = tc.zeros(n_features)
        if self.fit_intercept:
            self.intercept_ = y.mean()
        else:
            self.intercept_ = tc.zeros(1)
        
        # Coordinate descent algorithm
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                # Compute residual without feature j
                residual = y - X @ self.coef_ - self.intercept_
                
                # Update coefficient j
                X_j = X[:, j]
                rho = (residual * X_j).sum()
                norm = (X_j * X_j).sum()
                
                if norm > 0:
                    coef_j = rho / norm
                    soft_threshold = self.alpha / norm
                    
                    if coef_j > soft_threshold:
                        self.coef_[j] = coef_j - soft_threshold
                    elif coef_j < -soft_threshold:
                        self.coef_[j] = coef_j + soft_threshold
                    else:
                        self.coef_[j] = 0.0
            
            # Check convergence
            coef_diff = self.coef_ - coef_old
            max_change = coef_diff.abs().max()
            if max_change < self.tol:
                break
        
        self.fitted_ = True
        return self
    
    def _predict(self, X):
        """Make predictions."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        # Normalize features if they were normalized during training
        if self.normalize:
            X = (X - self.X_mean_) / self.X_std_
        
        predictions = X @ self.coef_
        if self.fit_intercept:
            predictions = predictions + self.intercept_
        
        return predictions


class ElasticNet(BaseLinearModel):
    """
    Elastic Net regression with L1 and L2 regularization.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Balance between L1 and L2 regularization.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    normalize : bool, default=False
        Whether to normalize the regressors.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    """
    
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, 
                 max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        """Fit elastic net regression model."""
        # Convert to TensorCore tensors if needed
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        n_samples, n_features = X.shape
        
        # Normalize features if requested
        if self.normalize:
            self.X_mean_ = X.mean(axis=0)
            self.X_std_ = X.std(axis=0)
            X = (X - self.X_mean_) / self.X_std_
        
        # Initialize coefficients
        self.coef_ = tc.zeros(n_features)
        if self.fit_intercept:
            self.intercept_ = y.mean()
        else:
            self.intercept_ = tc.zeros(1)
        
        # Coordinate descent with elastic net penalty
        alpha_l1 = self.alpha * self.l1_ratio
        alpha_l2 = self.alpha * (1.0 - self.l1_ratio)
        
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            
            for j in range(n_features):
                # Compute residual without feature j
                residual = y - X @ self.coef_ - self.intercept_
                
                # Update coefficient j
                X_j = X[:, j]
                rho = (residual * X_j).sum()
                norm = (X_j * X_j).sum()
                
                if norm > 0:
                    coef_j = rho / (norm + alpha_l2)
                    soft_threshold = alpha_l1 / (norm + alpha_l2)
                    
                    if coef_j > soft_threshold:
                        self.coef_[j] = coef_j - soft_threshold
                    elif coef_j < -soft_threshold:
                        self.coef_[j] = coef_j + soft_threshold
                    else:
                        self.coef_[j] = 0.0
            
            # Check convergence
            coef_diff = self.coef_ - coef_old
            max_change = coef_diff.abs().max()
            if max_change < self.tol:
                break
        
        self.fitted_ = True
        return self
    
    def _predict(self, X):
        """Make predictions."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        # Normalize features if they were normalized during training
        if self.normalize:
            X = (X - self.X_mean_) / self.X_std_
        
        predictions = X @ self.coef_
        if self.fit_intercept:
            predictions = predictions + self.intercept_
        
        return predictions


class LogisticRegression:
    """
    Logistic regression for binary and multiclass classification.
    
    Parameters
    ----------
    penalty : str, default='l2'
        Regularization penalty.
    C : float, default=1.0
        Inverse regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    multi_class : str, default='auto'
        Strategy for multiclass classification.
    """
    
    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, max_iter=1000, 
                 tol=1e-4, multi_class='auto'):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.multi_class = multi_class
        
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.fitted_ = False
    
    def fit(self, X, y):
        """Fit logistic regression model."""
        # Convert to TensorCore tensors if needed
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        # Get unique classes
        unique_classes = tc.unique(y)
        self.classes_ = unique_classes
        n_classes = len(unique_classes)
        n_samples, n_features = X.shape
        
        if n_classes == 2:
            # Binary classification
            y_binary = (y == unique_classes[1]).float()
            
            # Initialize coefficients
            self.coef_ = tc.zeros((1, n_features))
            if self.fit_intercept:
                self.intercept_ = tc.zeros(1)
            else:
                self.intercept_ = tc.zeros(1)
            
            # Gradient descent for logistic regression
            learning_rate = 0.01
            for iteration in range(self.max_iter):
                # Forward pass
                z = X @ self.coef_.T
                if self.fit_intercept:
                    z = z + self.intercept_
                
                # Sigmoid function
                sigmoid_z = 1.0 / (1.0 + tc.exp(-z))
                
                # Compute gradients
                error = sigmoid_z - y_binary.reshape(-1, 1)
                grad_coef = (X.T @ error / n_samples).T
                
                # Add regularization
                if self.penalty == 'l2':
                    grad_coef = grad_coef + (self.coef_ / self.C)
                
                # Update parameters
                self.coef_ = self.coef_ - learning_rate * grad_coef
                if self.fit_intercept:
                    grad_intercept = error.mean()
                    self.intercept_ = self.intercept_ - learning_rate * grad_intercept
        else:
            # Multiclass classification (One-vs-Rest)
            self.coef_ = tc.zeros((n_classes, n_features))
            self.intercept_ = tc.zeros(n_classes)
            
            for c in range(n_classes):
                # Create binary labels for class c
                y_binary = (y == unique_classes[c]).float()
                
                # Train binary classifier for class c
                coef_c = tc.zeros((1, n_features))
                intercept_c = tc.zeros(1)
                
                learning_rate = 0.01
                for iteration in range(self.max_iter):
                    z = X @ coef_c.T + intercept_c
                    sigmoid_z = 1.0 / (1.0 + tc.exp(-z))
                    
                    error = sigmoid_z - y_binary.reshape(-1, 1)
                    grad_coef = (X.T @ error / n_samples).T
                    
                    if self.penalty == 'l2':
                        grad_coef = grad_coef + (coef_c / self.C)
                    
                    coef_c = coef_c - learning_rate * grad_coef
                    grad_intercept = error.mean()
                    intercept_c = intercept_c - learning_rate * grad_intercept
                
                self.coef_[c] = coef_c[0]
                self.intercept_[c] = intercept_c[0]
        
        self.fitted_ = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification
            z = X @ self.coef_.T
            if self.fit_intercept:
                z = z + self.intercept_
            probabilities = 1.0 / (1.0 + tc.exp(-z))
            
            predictions = tc.zeros(n_samples)
            for i in range(n_samples):
                predictions[i] = self.classes_[1] if probabilities[i] > 0.5 else self.classes_[0]
            return predictions
        else:
            # Multiclass classification
            scores = X @ self.coef_.T
            if self.fit_intercept:
                scores = scores + self.intercept_
            
            predictions = tc.zeros(n_samples)
            for i in range(n_samples):
                max_class = 0
                max_score = scores[i, 0]
                for c in range(1, n_classes):
                    if scores[i, c] > max_score:
                        max_score = scores[i, c]
                        max_class = c
                predictions[i] = self.classes_[max_class]
            return predictions
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification
            z = X @ self.coef_.T
            if self.fit_intercept:
                z = z + self.intercept_
            probabilities = 1.0 / (1.0 + tc.exp(-z))
            
            proba = tc.zeros((n_samples, 2))
            for i in range(n_samples):
                p = probabilities[i]
                proba[i, 0] = 1.0 - p  # Probability of class 0
                proba[i, 1] = p        # Probability of class 1
            return proba
        else:
            # Multiclass classification
            scores = X @ self.coef_.T
            if self.fit_intercept:
                scores = scores + self.intercept_
            
            # Softmax
            proba = tc.zeros((n_samples, n_classes))
            for i in range(n_samples):
                row_scores = scores[i]
                exp_scores = tc.exp(row_scores - row_scores.max())
                softmax_scores = exp_scores / exp_scores.sum()
                proba[i] = softmax_scores
            return proba
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return tc.accuracy_score(y, y_pred)
