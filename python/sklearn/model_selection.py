"""
Model selection utilities for TensorCore

This module provides model selection and validation tools following the scikit-learn API.
"""

import tensorcore as tc
from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np
import random


def train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like
        Target values.
    test_size : float, default=0.25
        Proportion of dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    if not isinstance(X, tc.Tensor):
        X = tc.tensor(X)
    if not isinstance(y, tc.Tensor):
        y = tc.tensor(y)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Create indices
    indices = list(range(n_samples))
    if shuffle:
        random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Create train/test splits
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def cross_val_score(estimator, X, y, cv=5, scoring='accuracy', random_state=None):
    """
    Evaluate a score by cross-validation.
    
    Parameters
    ----------
    estimator : object
        Estimator object implementing 'fit' and 'predict' methods.
    X : array-like
        Input features.
    y : array-like
        Target values.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='accuracy'
        Scoring metric.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    array
        Array of scores for each fold.
    """
    if not isinstance(X, tc.Tensor):
        X = tc.tensor(X)
    if not isinstance(y, tc.Tensor):
        y = tc.tensor(y)
    
    n_samples = X.shape[0]
    fold_size = n_samples // cv
    
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    
    # Create indices
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    scores = []
    
    for i in range(cv):
        # Create fold indices
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < cv - 1 else n_samples
        
        test_indices = indices[start_idx:end_idx]
        train_indices = [idx for idx in indices if idx not in test_indices]
        
        # Split data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        # Train and evaluate
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        # Calculate score
        if scoring == 'accuracy':
            score = tc.accuracy_score(y_test, y_pred)
        elif scoring == 'precision':
            score = tc.precision_score(y_test, y_pred)
        elif scoring == 'recall':
            score = tc.recall_score(y_test, y_pred)
        elif scoring == 'f1':
            score = tc.f1_score(y_test, y_pred)
        elif scoring == 'r2':
            score = tc.r2_score(y_test, y_pred)
        elif scoring == 'mse':
            score = -tc.mean_squared_error(y_test, y_pred)  # Negative for maximization
        elif scoring == 'mae':
            score = -tc.mean_absolute_error(y_test, y_pred)  # Negative for maximization
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")
        
        scores.append(score)
    
    return tc.tensor(scores)


def KFold(n_splits=5, shuffle=False, random_state=None):
    """
    K-Fold cross-validator.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    KFold
        KFold cross-validator.
    """
    return _KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


class _KFold:
    """K-Fold cross-validator implementation."""
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like, optional
            Target values.
        
        Yields
        ------
        tuple
            (train_indices, test_indices)
        """
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        
        indices = list(range(n_samples))
        if self.shuffle:
            random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = [idx for idx in indices if idx not in test_indices]
            
            yield train_indices, test_indices


def StratifiedKFold(n_splits=5, shuffle=False, random_state=None):
    """
    Stratified K-Fold cross-validator.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    StratifiedKFold
        Stratified K-Fold cross-validator.
    """
    return _StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


class _StratifiedKFold:
    """Stratified K-Fold cross-validator implementation."""
    
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            Target values.
        
        Yields
        ------
        tuple
            (train_indices, test_indices)
        """
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Get unique classes and their counts
        classes = tc.unique(y)
        class_counts = {}
        class_indices = {}
        
        for class_val in classes:
            mask = (y == class_val)
            class_indices[class_val.item()] = [i for i, val in enumerate(mask) if val]
            class_counts[class_val.item()] = len(class_indices[class_val.item()])
        
        # Create stratified folds
        fold_indices = [[] for _ in range(self.n_splits)]
        
        for class_val, indices in class_indices.items():
            if self.shuffle:
                random.shuffle(indices)
            
            fold_size = len(indices) // self.n_splits
            remainder = len(indices) % self.n_splits
            
            start_idx = 0
            for i in range(self.n_splits):
                end_idx = start_idx + fold_size + (1 if i < remainder else 0)
                fold_indices[i].extend(indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Yield train/test splits
        for i in range(self.n_splits):
            test_indices = fold_indices[i]
            train_indices = []
            for j in range(self.n_splits):
                if j != i:
                    train_indices.extend(fold_indices[j])
            
            yield train_indices, test_indices


def GridSearchCV(estimator, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0):
    """
    Exhaustive search over specified parameter values for an estimator.
    
    Parameters
    ----------
    estimator : object
        Estimator object implementing 'fit' and 'predict' methods.
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings to try.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='accuracy'
        Scoring metric.
    n_jobs : int, default=1
        Number of jobs to run in parallel.
    verbose : int, default=0
        Verbosity level.
    
    Returns
    -------
    GridSearchCV
        Grid search cross-validator.
    """
    return _GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, 
                        scoring=scoring, n_jobs=n_jobs, verbose=verbose)


class _GridSearchCV:
    """Grid search cross-validator implementation."""
    
    def __init__(self, estimator, param_grid, cv=5, scoring='accuracy', n_jobs=1, verbose=0):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None
    
    def fit(self, X, y):
        """
        Fit the grid search to the data.
        
        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like
            Target values.
        
        Returns
        -------
        self
        """
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations()
        
        best_score = -float('inf')
        best_params = None
        best_estimator = None
        
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': []
        }
        
        for i, params in enumerate(param_combinations):
            if self.verbose > 0:
                print(f"Testing parameters {i+1}/{len(param_combinations)}: {params}")
            
            # Create estimator with current parameters
            estimator = self._create_estimator(params)
            
            # Perform cross-validation
            scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = scores.mean().item()
            std_score = scores.std().item()
            
            results['params'].append(params)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_estimator = estimator
        
        # Rank results
        sorted_indices = sorted(range(len(results['mean_test_score'])), 
                               key=lambda i: results['mean_test_score'][i], reverse=True)
        results['rank_test_score'] = [sorted_indices.index(i) + 1 for i in range(len(sorted_indices))]
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_estimator
        self.cv_results_ = results
        
        return self
    
    def _generate_param_combinations(self):
        """Generate all parameter combinations."""
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        combinations = []
        
        def generate_recursive(current_params, param_idx):
            if param_idx == len(param_names):
                combinations.append(current_params.copy())
                return
            
            param_name = param_names[param_idx]
            param_value_list = param_values[param_idx]
            
            for value in param_value_list:
                current_params[param_name] = value
                generate_recursive(current_params, param_idx + 1)
        
        generate_recursive({}, 0)
        return combinations
    
    def _create_estimator(self, params):
        """Create estimator with given parameters."""
        # This is a simplified implementation
        # In practice, you'd need to handle different estimator types
        estimator = self.estimator.__class__()
        for param_name, param_value in params.items():
            setattr(estimator, param_name, param_value)
        return estimator
    
    def predict(self, X):
        """Make predictions using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV must be fitted before making predictions")
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        """Return the score of the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV must be fitted before scoring")
        return self.best_estimator_.score(X, y)
