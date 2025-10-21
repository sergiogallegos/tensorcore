"""
Evaluation metrics for TensorCore

This module provides various evaluation metrics for machine learning models
following the scikit-learn API.
"""

import tensorcore as tc
from typing import Optional, Union, List
import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Accuracy classification score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    
    Returns
    -------
    float
        Accuracy score.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    correct = (y_true == y_pred).sum()
    return correct.item() / y_true.shape[0]


def precision_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Precision score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='binary'
        Averaging strategy.
    pos_label : int, default=1
        Label of positive class.
    
    Returns
    -------
    float
        Precision score.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    if average != 'binary':
        raise ValueError("Only binary average is currently supported")
    
    true_positives = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    false_positives = ((y_true != pos_label) & (y_pred == pos_label)).sum()
    
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives.item() / (true_positives + false_positives).item()


def recall_score(y_true, y_pred, average='binary', pos_label=1):
    """
    Recall score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='binary'
        Averaging strategy.
    pos_label : int, default=1
        Label of positive class.
    
    Returns
    -------
    float
        Recall score.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    if average != 'binary':
        raise ValueError("Only binary average is currently supported")
    
    true_positives = ((y_true == pos_label) & (y_pred == pos_label)).sum()
    false_negatives = ((y_true == pos_label) & (y_pred != pos_label)).sum()
    
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives.item() / (true_positives + false_negatives).item()


def f1_score(y_true, y_pred, average='binary', pos_label=1):
    """
    F1 score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='binary'
        Averaging strategy.
    pos_label : int, default=1
        Label of positive class.
    
    Returns
    -------
    float
        F1 score.
    """
    precision = precision_score(y_true, y_pred, average, pos_label)
    recall = recall_score(y_true, y_pred, average, pos_label)
    
    if precision + recall == 0.0:
        return 0.0
    
    return 2.0 * precision * recall / (precision + recall)


def roc_auc_score(y_true, y_score, average='macro'):
    """
    ROC AUC score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_score : array-like
        Predicted scores or probabilities.
    average : str, default='macro'
        Averaging strategy.
    
    Returns
    -------
    float
        ROC AUC score.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_score, tc.Tensor):
        y_score = tc.tensor(y_score)
    
    # Simplified implementation for binary classification
    if y_score.shape[1] == 2:
        y_score = y_score[:, 1]  # Use positive class probabilities
    
    # Sort by scores
    sorted_indices = y_score.argsort(descending=True)
    y_true_sorted = y_true[sorted_indices]
    
    # Calculate ROC AUC
    n_positive = (y_true == 1).sum()
    n_negative = (y_true == 0).sum()
    
    if n_positive == 0 or n_negative == 0:
        return 0.5
    
    tp = 0
    fp = 0
    auc = 0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    
    return auc / (n_positive * n_negative)


def mean_squared_error(y_true, y_pred):
    """
    Mean squared error.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    float
        Mean squared error.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    residuals = y_true - y_pred
    return (residuals * residuals).mean().item()


def mean_absolute_error(y_true, y_pred):
    """
    Mean absolute error.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    float
        Mean absolute error.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    residuals = y_true - y_pred
    return residuals.abs().mean().item()


def r2_score(y_true, y_pred):
    """
    R^2 (coefficient of determination) score.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.
    
    Returns
    -------
    float
        R^2 score.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    ss_res = ((y_true - y_pred) * (y_true - y_pred)).sum()
    ss_tot = ((y_true - y_true.mean()) * (y_true - y_true.mean())).sum()
    
    if ss_tot == 0:
        return 0.0
    
    return 1.0 - (ss_res.item() / ss_tot.item())


def classification_report(y_true, y_pred, target_names=None):
    """
    Build a text report showing the main classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    target_names : list, optional
        Names of the classes.
    
    Returns
    -------
    str
        Classification report.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    # Get unique classes
    classes = tc.unique(tc.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = [f"class_{i}" for i in classes]
    
    # Calculate metrics for each class
    report = "Classification Report\n"
    report += "=" * 50 + "\n"
    report += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
    report += "-" * 50 + "\n"
    
    total_support = 0
    for i, class_name in enumerate(target_names):
        if i < len(classes):
            class_label = classes[i]
            
            # Calculate metrics for this class
            y_true_binary = (y_true == class_label).float()
            y_pred_binary = (y_pred == class_label).float()
            
            tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum().item()
            fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum().item()
            fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp + fn
            
            report += f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {support:<10.0f}\n"
            total_support += support
    
    report += "-" * 50 + "\n"
    report += f"{'Total':<15} {'':<10} {'':<10} {'':<10} {total_support:<10.0f}\n"
    
    return report


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        List of labels to index the matrix.
    
    Returns
    -------
    array
        Confusion matrix.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    if labels is None:
        labels = tc.unique(tc.concatenate([y_true, y_pred]))
    
    n_classes = len(labels)
    matrix = tc.zeros((n_classes, n_classes))
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = ((y_true == true_label) & (y_pred == pred_label)).sum()
            matrix[i, j] = count
    
    return matrix


def classification_report_dict(y_true, y_pred, target_names=None):
    """
    Build a dictionary report showing the main classification metrics.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    target_names : list, optional
        Names of the classes.
    
    Returns
    -------
    dict
        Classification report dictionary.
    """
    if not isinstance(y_true, tc.Tensor):
        y_true = tc.tensor(y_true)
    if not isinstance(y_pred, tc.Tensor):
        y_pred = tc.tensor(y_pred)
    
    # Get unique classes
    classes = tc.unique(tc.concatenate([y_true, y_pred]))
    
    if target_names is None:
        target_names = [f"class_{i}" for i in classes]
    
    report = {}
    
    for i, class_name in enumerate(target_names):
        if i < len(classes):
            class_label = classes[i]
            
            # Calculate metrics for this class
            y_true_binary = (y_true == class_label).float()
            y_pred_binary = (y_pred == class_label).float()
            
            tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum().item()
            fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum().item()
            fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp + fn
            
            report[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': support
            }
    
    return report
