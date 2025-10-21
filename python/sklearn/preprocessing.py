"""
Preprocessing utilities for TensorCore

This module provides data preprocessing tools following the scikit-learn API.
"""

import tensorcore as tc
from typing import Optional, Union, Tuple
import numpy as np


class BaseScaler:
    """Base class for scalers."""
    
    def __init__(self):
        self.fitted_ = False
    
    def fit(self, X, y=None):
        """Fit the scaler to the data."""
        raise NotImplementedError
    
    def transform(self, X):
        """Transform the data."""
        if not self.fitted_:
            raise ValueError("Scaler must be fitted before transforming")
        return self._transform(X)
    
    def _transform(self, X):
        """Internal transform method."""
        raise NotImplementedError
    
    def fit_transform(self, X, y=None):
        """Fit the scaler and transform the data."""
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform the data."""
        raise NotImplementedError


class StandardScaler(BaseScaler):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Parameters
    ----------
    with_mean : bool, default=True
        Whether to center the data.
    with_std : bool, default=True
        Whether to scale to unit variance.
    copy : bool, default=True
        Whether to copy the data.
    """
    
    def __init__(self, with_mean=True, with_std=True, copy=True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X, y=None):
        """Fit the scaler to the data."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        if self.with_mean:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = tc.zeros(X.shape[1])
        
        if self.with_std:
            self.scale_ = X.std(axis=0)
            # Avoid division by zero
            self.scale_ = tc.where(self.scale_ == 0, tc.ones_like(self.scale_), self.scale_)
        else:
            self.scale_ = tc.ones(X.shape[1])
        
        self.fitted_ = True
        return self
    
    def _transform(self, X):
        """Transform the data."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        X_scaled = X - self.mean_
        X_scaled = X_scaled / self.scale_
        
        return X_scaled
    
    def inverse_transform(self, X):
        """Inverse transform the data."""
        if not self.fitted_:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        X_original = X * self.scale_
        X_original = X_original + self.mean_
        
        return X_original


class MinMaxScaler(BaseScaler):
    """
    Transform features by scaling each feature to a given range.
    
    Parameters
    ----------
    feature_range : tuple, default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Whether to copy the data.
    """
    
    def __init__(self, feature_range=(0, 1), copy=True):
        super().__init__()
        self.feature_range = feature_range
        self.copy = copy
        self.min_ = None
        self.scale_ = None
    
    def fit(self, X, y=None):
        """Fit the scaler to the data."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        self.min_ = X.min(axis=0)
        data_range = X.max(axis=0) - self.min_
        
        # Avoid division by zero
        data_range = tc.where(data_range == 0, tc.ones_like(data_range), data_range)
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        
        self.fitted_ = True
        return self
    
    def _transform(self, X):
        """Transform the data."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        X_scaled = X - self.min_
        X_scaled = X_scaled * self.scale_
        X_scaled = X_scaled + self.feature_range[0]
        
        return X_scaled
    
    def inverse_transform(self, X):
        """Inverse transform the data."""
        if not self.fitted_:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        X_original = X - self.feature_range[0]
        X_original = X_original / self.scale_
        X_original = X_original + self.min_
        
        return X_original


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    """
    
    def __init__(self):
        self.classes_ = None
        self.fitted_ = False
    
    def fit(self, y, X=None):
        """Fit the label encoder."""
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        # Get unique classes
        unique_classes = tc.unique(y)
        self.classes_ = unique_classes
        
        self.fitted_ = True
        return self
    
    def transform(self, y):
        """Transform labels to normalized encoding."""
        if not self.fitted_:
            raise ValueError("LabelEncoder must be fitted before transforming")
        
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        y_encoded = tc.zeros_like(y)
        for i, class_val in enumerate(self.classes_):
            mask = (y == class_val)
            y_encoded = tc.where(mask, i, y_encoded)
        
        return y_encoded
    
    def fit_transform(self, y, X=None):
        """Fit the label encoder and transform labels."""
        self.fit(y, X)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """Transform labels back to original encoding."""
        if not self.fitted_:
            raise ValueError("LabelEncoder must be fitted before inverse transforming")
        
        if not isinstance(y, tc.Tensor):
            y = tc.tensor(y)
        
        y_original = tc.zeros_like(y)
        for i, class_val in enumerate(self.classes_):
            mask = (y == i)
            y_original = tc.where(mask, class_val, y_original)
        
        return y_original


class OneHotEncoder:
    """
    Encode categorical integer features as a one-hot numeric array.
    
    Parameters
    ----------
    categories : str or list, default='auto'
        Categories for each feature.
    drop : str, default='if_binary'
        Whether to drop one category.
    sparse : bool, default=False
        Whether to return sparse matrix.
    dtype : str, default='float'
        Data type of output.
    """
    
    def __init__(self, categories='auto', drop='if_binary', sparse=False, dtype='float'):
        self.categories = categories
        self.drop = drop
        self.sparse = sparse
        self.dtype = dtype
        self.categories_ = None
        self.fitted_ = False
    
    def fit(self, X, y=None):
        """Fit the one-hot encoder."""
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        n_features = X.shape[1]
        self.categories_ = []
        
        for i in range(n_features):
            feature_values = X[:, i]
            unique_values = tc.unique(feature_values)
            self.categories_.append(unique_values)
        
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """Transform categorical data to one-hot encoding."""
        if not self.fitted_:
            raise ValueError("OneHotEncoder must be fitted before transforming")
        
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        n_samples, n_features = X.shape
        n_categories = sum(len(cats) for cats in self.categories_)
        
        # Create one-hot encoding
        X_encoded = tc.zeros((n_samples, n_categories))
        col_idx = 0
        
        for i in range(n_features):
            feature_values = X[:, i]
            categories = self.categories_[i]
            
            for j, category in enumerate(categories):
                mask = (feature_values == category)
                X_encoded[:, col_idx + j] = mask.float()
            
            col_idx += len(categories)
        
        return X_encoded
    
    def fit_transform(self, X, y=None):
        """Fit the one-hot encoder and transform data."""
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Transform one-hot encoding back to categorical data."""
        if not self.fitted_:
            raise ValueError("OneHotEncoder must be fitted before inverse transforming")
        
        if not isinstance(X, tc.Tensor):
            X = tc.tensor(X)
        
        n_samples = X.shape[0]
        n_features = len(self.categories_)
        
        X_original = tc.zeros((n_samples, n_features))
        col_idx = 0
        
        for i in range(n_features):
            categories = self.categories_[i]
            n_cats = len(categories)
            
            # Find the category with maximum value (one-hot)
            max_indices = X[:, col_idx:col_idx + n_cats].argmax(axis=1)
            
            for j in range(n_samples):
                X_original[j, i] = categories[max_indices[j]]
            
            col_idx += n_cats
        
        return X_original
