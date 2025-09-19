"""
Neural network layers for TensorCore

This module provides various neural network layers commonly used in deep learning.
"""

from ..tensorcore_core import Tensor
from typing import List, Optional, Union, Callable
import math

class Layer:
    """Base class for all neural network layers."""
    
    def __init__(self):
        self.training = True
        self.parameters = []
        self.gradients = []
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer."""
        raise NotImplementedError
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through the layer."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out all gradients."""
        for grad in self.gradients:
            if grad is not None:
                grad.fill(0.0)
    
    def parameters(self) -> List[Tensor]:
        """Return list of learnable parameters."""
        return self.parameters
    
    def gradients(self) -> List[Tensor]:
        """Return list of parameter gradients."""
        return self.gradients
    
    def train(self):
        """Set layer to training mode."""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False

class Dense(Layer):
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size: int, output_size: int, use_bias: bool = True,
                 activation: Optional[Callable] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.activation = activation
        
        # Initialize weights using Xavier/Glorot initialization
        fan_in = input_size
        fan_out = output_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        
        self.weight = Tensor.random_uniform(
            (input_size, output_size), 
            -limit, limit
        )
        self.weight.set_requires_grad(True)
        self.parameters.append(self.weight)
        
        if use_bias:
            self.bias = Tensor.zeros((output_size,))
            self.bias.set_requires_grad(True)
            self.parameters.append(self.bias)
        else:
            self.bias = None
        
        # Initialize gradients
        self.weight_grad = Tensor.zeros_like(self.weight)
        self.bias_grad = Tensor.zeros_like(self.bias) if use_bias else None
        self.gradients.extend([self.weight_grad, self.bias_grad])
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = xW + b"""
        self.last_input = x
        
        # Matrix multiplication: x @ weight
        output = x.matmul(self.weight)
        
        # Add bias if enabled
        if self.use_bias:
            output = output + self.bias
        
        # Apply activation function if provided
        if self.activation is not None:
            output = self.activation(output)
        
        self.last_output = output
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass: compute gradients"""
        # Gradient w.r.t. input
        grad_input = grad_output.matmul(self.weight.transpose())
        
        # Gradient w.r.t. weights
        self.weight_grad = self.last_input.transpose().matmul(grad_output)
        
        # Gradient w.r.t. bias
        if self.use_bias:
            self.bias_grad = grad_output.sum(axis=0)
        
        return grad_input

class Conv2D(Layer):
    """2D Convolutional layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, use_bias: bool = True,
                 activation: Optional[Callable] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation
        
        # Initialize weights using Xavier/Glorot initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        
        self.weight = Tensor.random_uniform(
            (out_channels, in_channels, kernel_size, kernel_size),
            -limit, limit
        )
        self.weight.set_requires_grad(True)
        self.parameters.append(self.weight)
        
        if use_bias:
            self.bias = Tensor.zeros((out_channels,))
            self.bias.set_requires_grad(True)
            self.parameters.append(self.bias)
        else:
            self.bias = None
        
        # Initialize gradients
        self.weight_grad = Tensor.zeros_like(self.weight)
        self.bias_grad = Tensor.zeros_like(self.bias) if use_bias else None
        self.gradients.extend([self.weight_grad, self.bias_grad])
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: 2D convolution"""
        self.last_input = x
        
        # TODO: Implement 2D convolution
        # This is a placeholder implementation
        batch_size, in_channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = Tensor.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Add bias if enabled
        if self.use_bias:
            for i in range(self.out_channels):
                output[:, i, :, :] = output[:, i, :, :] + self.bias[i]
        
        # Apply activation function if provided
        if self.activation is not None:
            output = self.activation(output)
        
        self.last_output = output
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass: compute gradients"""
        # TODO: Implement 2D convolution backward pass
        # This is a placeholder implementation
        grad_input = Tensor.zeros_like(self.last_input)
        return grad_input

class MaxPool2D(Layer):
    """2D Max Pooling layer."""
    
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: 2D max pooling"""
        self.last_input = x
        
        # TODO: Implement 2D max pooling
        # This is a placeholder implementation
        batch_size, channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = Tensor.zeros((batch_size, channels, out_height, out_width))
        self.last_output = output
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass: compute gradients"""
        # TODO: Implement 2D max pooling backward pass
        # This is a placeholder implementation
        grad_input = Tensor.zeros_like(self.last_input)
        return grad_input

class Dropout(Layer):
    """Dropout layer for regularization."""
    
    def __init__(self, rate: float = 0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: apply dropout during training"""
        self.last_input = x
        
        if self.training and self.rate > 0:
            # Create dropout mask
            self.mask = Tensor.random_uniform(x.shape, 0, 1) > self.rate
            output = x * self.mask / (1 - self.rate)
        else:
            output = x
        
        self.last_output = output
        return output
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass: apply dropout mask to gradients"""
        if self.training and self.rate > 0:
            grad_input = grad_output * self.mask / (1 - self.rate)
        else:
            grad_input = grad_output
        
        return grad_input

class Sequential(Layer):
    """Sequential container for layers."""
    
    def __init__(self, layers: Optional[List[Layer]] = None):
        super().__init__()
        self.layers = layers or []
        self._update_parameters()
    
    def add(self, layer: Layer):
        """Add a layer to the sequential container."""
        self.layers.append(layer)
        self._update_parameters()
    
    def _update_parameters(self):
        """Update the list of parameters from all layers."""
        self.parameters = []
        self.gradients = []
        for layer in self.layers:
            self.parameters.extend(layer.parameters)
            self.gradients.extend(layer.gradients)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output: Tensor) -> Tensor:
        """Backward pass through all layers in reverse order."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def zero_grad(self):
        """Zero out all gradients in all layers."""
        for layer in self.layers:
            layer.zero_grad()
    
    def train(self):
        """Set all layers to training mode."""
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        """Set all layers to evaluation mode."""
        for layer in self.layers:
            layer.eval()
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, index):
        return self.layers[index]
