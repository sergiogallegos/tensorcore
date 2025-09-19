"""
Activation functions for TensorCore

This module provides various activation functions commonly used in neural networks.
"""

from ..tensorcore_core import Tensor
from typing import Callable
import math

class ActivationFunction:
    """Base class for activation functions."""
    
    def __init__(self, forward_func: Callable, backward_func: Callable):
        self.forward_func = forward_func
        self.backward_func = backward_func
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the activation function."""
        return self.forward_func(x)
    
    def backward(self, x: Tensor, grad_output: Tensor) -> Tensor:
        """Backward pass through the activation function."""
        return self.backward_func(x, grad_output)

def _relu_forward(x: Tensor) -> Tensor:
    """ReLU forward pass."""
    return x.maximum(0.0)

def _relu_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """ReLU backward pass."""
    return grad_output * (x > 0.0)

def _leaky_relu_forward(x: Tensor, alpha: float = 0.01) -> Tensor:
    """Leaky ReLU forward pass."""
    return x.maximum(alpha * x)

def _leaky_relu_backward(x: Tensor, grad_output: Tensor, alpha: float = 0.01) -> Tensor:
    """Leaky ReLU backward pass."""
    return grad_output * ((x > 0.0) + alpha * (x <= 0.0))

def _sigmoid_forward(x: Tensor) -> Tensor:
    """Sigmoid forward pass."""
    return 1.0 / (1.0 + (-x).exp())

def _sigmoid_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Sigmoid backward pass."""
    sigmoid_x = _sigmoid_forward(x)
    return grad_output * sigmoid_x * (1.0 - sigmoid_x)

def _tanh_forward(x: Tensor) -> Tensor:
    """Tanh forward pass."""
    return x.tanh()

def _tanh_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Tanh backward pass."""
    tanh_x = x.tanh()
    return grad_output * (1.0 - tanh_x * tanh_x)

def _softmax_forward(x: Tensor) -> Tensor:
    """Softmax forward pass."""
    # Subtract max for numerical stability
    x_max = x.max()
    x_shifted = x - x_max
    exp_x = x_shifted.exp()
    return exp_x / exp_x.sum()

def _softmax_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Softmax backward pass."""
    softmax_x = _softmax_forward(x)
    return grad_output * softmax_x * (1.0 - softmax_x)

def _gelu_forward(x: Tensor) -> Tensor:
    """GELU forward pass."""
    return 0.5 * x * (1.0 + ((2.0 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3)).tanh())

def _gelu_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """GELU backward pass."""
    # This is a simplified implementation
    # The actual GELU derivative is more complex
    return grad_output * (1.0 + ((2.0 / math.pi) ** 0.5 * (x + 0.044715 * x ** 3)).tanh())

def _swish_forward(x: Tensor, beta: float = 1.0) -> Tensor:
    """Swish forward pass."""
    return x * (beta * x).sigmoid()

def _swish_backward(x: Tensor, grad_output: Tensor, beta: float = 1.0) -> Tensor:
    """Swish backward pass."""
    sigmoid_beta_x = (beta * x).sigmoid()
    return grad_output * (sigmoid_beta_x + beta * x * sigmoid_beta_x * (1.0 - sigmoid_beta_x))

def _mish_forward(x: Tensor) -> Tensor:
    """Mish forward pass."""
    return x * (1.0 + x.exp()).log().tanh()

def _mish_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Mish backward pass."""
    # This is a simplified implementation
    # The actual Mish derivative is more complex
    exp_x = x.exp()
    return grad_output * (exp_x * (4.0 * (x + 1.0) + 4.0 * exp_x + exp_x * exp_x) / 
                         ((1.0 + exp_x) * (1.0 + exp_x)))

def _elu_forward(x: Tensor, alpha: float = 1.0) -> Tensor:
    """ELU forward pass."""
    return x.maximum(0.0) + alpha * ((x.minimum(0.0)).exp() - 1.0)

def _elu_backward(x: Tensor, grad_output: Tensor, alpha: float = 1.0) -> Tensor:
    """ELU backward pass."""
    return grad_output * ((x > 0.0) + alpha * (x <= 0.0) * x.exp())

# Predefined activation functions
ReLU = ActivationFunction(_relu_forward, _relu_backward)
LeakyReLU = ActivationFunction(
    lambda x: _leaky_relu_forward(x, 0.01),
    lambda x, grad: _leaky_relu_backward(x, grad, 0.01)
)
Sigmoid = ActivationFunction(_sigmoid_forward, _sigmoid_backward)
Tanh = ActivationFunction(_tanh_forward, _tanh_backward)
Softmax = ActivationFunction(_softmax_forward, _softmax_backward)
GELU = ActivationFunction(_gelu_forward, _gelu_backward)
Swish = ActivationFunction(
    lambda x: _swish_forward(x, 1.0),
    lambda x, grad: _swish_backward(x, grad, 1.0)
)
Mish = ActivationFunction(_mish_forward, _mish_backward)
ELU = ActivationFunction(
    lambda x: _elu_forward(x, 1.0),
    lambda x, grad: _elu_backward(x, grad, 1.0)
)

# Additional activation functions
def _hard_sigmoid_forward(x: Tensor) -> Tensor:
    """Hard sigmoid forward pass."""
    return (0.2 * x + 0.5).clamp(0.0, 1.0)

def _hard_sigmoid_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Hard sigmoid backward pass."""
    return grad_output * ((x > -2.5) & (x < 2.5)) * 0.2

def _hard_tanh_forward(x: Tensor) -> Tensor:
    """Hard tanh forward pass."""
    return x.clamp(-1.0, 1.0)

def _hard_tanh_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Hard tanh backward pass."""
    return grad_output * ((x > -1.0) & (x < 1.0))

def _softplus_forward(x: Tensor) -> Tensor:
    """Softplus forward pass."""
    return (1.0 + x.exp()).log()

def _softplus_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Softplus backward pass."""
    return grad_output * (1.0 / (1.0 + (-x).exp()))

def _softsign_forward(x: Tensor) -> Tensor:
    """Softsign forward pass."""
    return x / (1.0 + x.abs())

def _softsign_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Softsign backward pass."""
    return grad_output / ((1.0 + x.abs()) ** 2)

def _identity_forward(x: Tensor) -> Tensor:
    """Identity forward pass."""
    return x

def _identity_backward(x: Tensor, grad_output: Tensor) -> Tensor:
    """Identity backward pass."""
    return grad_output

# Additional predefined activation functions
HardSigmoid = ActivationFunction(_hard_sigmoid_forward, _hard_sigmoid_backward)
HardTanh = ActivationFunction(_hard_tanh_forward, _hard_tanh_backward)
Softplus = ActivationFunction(_softplus_forward, _softplus_backward)
Softsign = ActivationFunction(_softsign_forward, _softsign_backward)
Identity = ActivationFunction(_identity_forward, _identity_backward)

# Activation function registry
ACTIVATION_FUNCTIONS = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'softmax': Softmax,
    'gelu': GELU,
    'swish': Swish,
    'mish': Mish,
    'elu': ELU,
    'hard_sigmoid': HardSigmoid,
    'hard_tanh': HardTanh,
    'softplus': Softplus,
    'softsign': Softsign,
    'identity': Identity,
}

def get_activation(name: str) -> ActivationFunction:
    """Get activation function by name."""
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    return ACTIVATION_FUNCTIONS[name]

def list_activations() -> list:
    """List all available activation functions."""
    return list(ACTIVATION_FUNCTIONS.keys())
