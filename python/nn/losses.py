"""
Loss functions for TensorCore

This module provides various loss functions commonly used in machine learning.
"""

from ..tensorcore_core import Tensor
from typing import Callable, Optional
import math

class LossFunction:
    """Base class for loss functions."""
    
    def __init__(self, forward_func: Callable, backward_func: Callable):
        self.forward_func = forward_func
        self.backward_func = backward_func
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass through the loss function."""
        return self.forward_func(predictions, targets)
    
    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Backward pass through the loss function."""
        return self.backward_func(predictions, targets)

def _mse_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """MSE forward pass."""
    return ((predictions - targets) ** 2).mean()

def _mse_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """MSE backward pass."""
    return 2.0 * (predictions - targets) / predictions.size

def _mae_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """MAE forward pass."""
    return (predictions - targets).abs().mean()

def _mae_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """MAE backward pass."""
    diff = predictions - targets
    return diff.sign() / predictions.size

def _huber_forward(predictions: Tensor, targets: Tensor, delta: float = 1.0) -> Tensor:
    """Huber loss forward pass."""
    diff = predictions - targets
    abs_diff = diff.abs()
    return ((abs_diff <= delta) * 0.5 * diff ** 2 + 
            (abs_diff > delta) * delta * (abs_diff - 0.5 * delta)).mean()

def _huber_backward(predictions: Tensor, targets: Tensor, delta: float = 1.0) -> Tensor:
    """Huber loss backward pass."""
    diff = predictions - targets
    abs_diff = diff.abs()
    return ((abs_diff <= delta) * diff + 
            (abs_diff > delta) * delta * diff.sign()) / predictions.size

def _cross_entropy_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy forward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Compute cross-entropy loss
    log_pred = softmax_pred.log()
    return -(targets * log_pred).sum(axis=-1).mean()

def _cross_entropy_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy backward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Compute gradient
    return (softmax_pred - targets) / predictions.size

def _binary_cross_entropy_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy forward pass."""
    # Apply sigmoid to predictions
    sigmoid_pred = 1.0 / (1.0 + (-predictions).exp())
    
    # Compute binary cross-entropy loss
    log_pred = sigmoid_pred.log()
    log_pred_neg = (1.0 - sigmoid_pred).log()
    return -(targets * log_pred + (1.0 - targets) * log_pred_neg).mean()

def _binary_cross_entropy_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy backward pass."""
    # Apply sigmoid to predictions
    sigmoid_pred = 1.0 / (1.0 + (-predictions).exp())
    
    # Compute gradient
    return (sigmoid_pred - targets) / predictions.size

def _categorical_cross_entropy_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Categorical cross-entropy forward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Compute categorical cross-entropy loss
    log_pred = softmax_pred.log()
    return -(targets * log_pred).sum(axis=-1).mean()

def _categorical_cross_entropy_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Categorical cross-entropy backward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Compute gradient
    return (softmax_pred - targets) / predictions.size

def _sparse_categorical_cross_entropy_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Sparse categorical cross-entropy forward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Convert targets to one-hot encoding
    batch_size = targets.size
    num_classes = predictions.shape[-1]
    targets_one_hot = Tensor.zeros((batch_size, num_classes))
    for i in range(batch_size):
        targets_one_hot[i, int(targets[i])] = 1.0
    
    # Compute sparse categorical cross-entropy loss
    log_pred = softmax_pred.log()
    return -(targets_one_hot * log_pred).sum(axis=-1).mean()

def _sparse_categorical_cross_entropy_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Sparse categorical cross-entropy backward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Convert targets to one-hot encoding
    batch_size = targets.size
    num_classes = predictions.shape[-1]
    targets_one_hot = Tensor.zeros((batch_size, num_classes))
    for i in range(batch_size):
        targets_one_hot[i, int(targets[i])] = 1.0
    
    # Compute gradient
    return (softmax_pred - targets_one_hot) / predictions.size

def _hinge_forward(predictions: Tensor, targets: Tensor, margin: float = 1.0) -> Tensor:
    """Hinge loss forward pass."""
    # Convert targets to -1, 1 format
    targets_binary = 2.0 * targets - 1.0
    
    # Compute hinge loss
    loss = (1.0 - targets_binary * predictions).maximum(0.0)
    return loss.mean()

def _hinge_backward(predictions: Tensor, targets: Tensor, margin: float = 1.0) -> Tensor:
    """Hinge loss backward pass."""
    # Convert targets to -1, 1 format
    targets_binary = 2.0 * targets - 1.0
    
    # Compute gradient
    grad = -targets_binary * (1.0 - targets_binary * predictions > 0.0)
    return grad / predictions.size

def _squared_hinge_forward(predictions: Tensor, targets: Tensor, margin: float = 1.0) -> Tensor:
    """Squared hinge loss forward pass."""
    # Convert targets to -1, 1 format
    targets_binary = 2.0 * targets - 1.0
    
    # Compute squared hinge loss
    loss = ((1.0 - targets_binary * predictions).maximum(0.0)) ** 2
    return loss.mean()

def _squared_hinge_backward(predictions: Tensor, targets: Tensor, margin: float = 1.0) -> Tensor:
    """Squared hinge loss backward pass."""
    # Convert targets to -1, 1 format
    targets_binary = 2.0 * targets - 1.0
    
    # Compute gradient
    grad = -2.0 * targets_binary * (1.0 - targets_binary * predictions).maximum(0.0)
    return grad / predictions.size

def _kl_divergence_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """KL divergence forward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Apply softmax to targets
    exp_targets = targets.exp()
    softmax_targets = exp_targets / exp_targets.sum(axis=-1, keepdims=True)
    
    # Compute KL divergence
    log_pred = softmax_pred.log()
    log_targets = softmax_targets.log()
    return (softmax_pred * (log_pred - log_targets)).sum(axis=-1).mean()

def _kl_divergence_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """KL divergence backward pass."""
    # Apply softmax to predictions
    exp_pred = predictions.exp()
    softmax_pred = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    
    # Apply softmax to targets
    exp_targets = targets.exp()
    softmax_targets = exp_targets / exp_targets.sum(axis=-1, keepdims=True)
    
    # Compute gradient
    return (softmax_pred - softmax_targets) / predictions.size

# Predefined loss functions
MSELoss = LossFunction(_mse_forward, _mse_backward)
MAELoss = LossFunction(_mae_forward, _mae_backward)
HuberLoss = LossFunction(
    lambda p, t: _huber_forward(p, t, 1.0),
    lambda p, t: _huber_backward(p, t, 1.0)
)
CrossEntropyLoss = LossFunction(_cross_entropy_forward, _cross_entropy_backward)
BinaryCrossEntropyLoss = LossFunction(_binary_cross_entropy_forward, _binary_cross_entropy_backward)
CategoricalCrossEntropyLoss = LossFunction(_categorical_cross_entropy_forward, _categorical_cross_entropy_backward)
SparseCategoricalCrossEntropyLoss = LossFunction(_sparse_categorical_cross_entropy_forward, _sparse_categorical_cross_entropy_backward)
HingeLoss = LossFunction(
    lambda p, t: _hinge_forward(p, t, 1.0),
    lambda p, t: _hinge_backward(p, t, 1.0)
)
SquaredHingeLoss = LossFunction(
    lambda p, t: _squared_hinge_forward(p, t, 1.0),
    lambda p, t: _squared_hinge_backward(p, t, 1.0)
)
KLDivergenceLoss = LossFunction(_kl_divergence_forward, _kl_divergence_backward)

# Additional loss functions
def _poisson_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Poisson loss forward pass."""
    return (predictions - targets * predictions.log()).mean()

def _poisson_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Poisson loss backward pass."""
    return (1.0 - targets / predictions) / predictions.size

def _cosine_similarity_forward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cosine similarity loss forward pass."""
    # Normalize predictions and targets
    pred_norm = predictions / predictions.norm(axis=-1, keepdims=True)
    target_norm = targets / targets.norm(axis=-1, keepdims=True)
    
    # Compute cosine similarity
    cosine_sim = (pred_norm * target_norm).sum(axis=-1)
    return (1.0 - cosine_sim).mean()

def _cosine_similarity_backward(predictions: Tensor, targets: Tensor) -> Tensor:
    """Cosine similarity loss backward pass."""
    # Normalize predictions and targets
    pred_norm = predictions / predictions.norm(axis=-1, keepdims=True)
    target_norm = targets / targets.norm(axis=-1, keepdims=True)
    
    # Compute gradient
    cosine_sim = (pred_norm * target_norm).sum(axis=-1, keepdims=True)
    grad = (target_norm - cosine_sim * pred_norm) / predictions.norm(axis=-1, keepdims=True)
    return -grad / predictions.size

# Additional predefined loss functions
PoissonLoss = LossFunction(_poisson_forward, _poisson_backward)
CosineSimilarityLoss = LossFunction(_cosine_similarity_forward, _cosine_similarity_backward)

# Loss function registry
LOSS_FUNCTIONS = {
    'mse': MSELoss,
    'mae': MAELoss,
    'huber': HuberLoss,
    'cross_entropy': CrossEntropyLoss,
    'binary_cross_entropy': BinaryCrossEntropyLoss,
    'categorical_cross_entropy': CategoricalCrossEntropyLoss,
    'sparse_categorical_cross_entropy': SparseCategoricalCrossEntropyLoss,
    'hinge': HingeLoss,
    'squared_hinge': SquaredHingeLoss,
    'kl_divergence': KLDivergenceLoss,
    'poisson': PoissonLoss,
    'cosine_similarity': CosineSimilarityLoss,
}

def get_loss(name: str) -> LossFunction:
    """Get loss function by name."""
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_FUNCTIONS[name]

def list_losses() -> list:
    """List all available loss functions."""
    return list(LOSS_FUNCTIONS.keys())
