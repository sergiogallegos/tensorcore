"""
Scikit-learn style machine learning algorithms for TensorCore

This module provides implementations of popular machine learning algorithms
following the scikit-learn API design patterns with fit/predict/transform methods.
"""

from .linear_model import *
from .tree import *
from .cluster import *
from .preprocessing import *
from .metrics import *
from .model_selection import *

__all__ = [
    # Linear models
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LogisticRegression',
    
    # Tree models
    'DecisionTreeClassifier', 'DecisionTreeRegressor',
    
    # Clustering
    'KMeans',
    
    # Preprocessing
    'StandardScaler', 'MinMaxScaler', 'LabelEncoder', 'OneHotEncoder',
    
    # Metrics
    'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
    'mean_squared_error', 'mean_absolute_error', 'r2_score',
    
    # Model selection
    'train_test_split', 'cross_val_score',
]
