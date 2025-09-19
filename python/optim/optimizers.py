"""
Optimization algorithms for TensorCore

This module provides various optimization algorithms commonly used in machine learning.
"""

from ..tensorcore_core import Tensor
from typing import List, Optional, Dict, Any
import math

class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr
        self.step_count = 0
    
    def step(self):
        """Perform one optimization step."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero out all gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.fill(0.0)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.lr
    
    def set_lr(self, lr: float):
        """Set learning rate."""
        self.lr = lr

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, 
                 momentum: float = 0.0, dampening: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        # Initialize velocity vectors
        self.velocities = []
        for param in parameters:
            self.velocities.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one SGD step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad = param.grad + self.weight_decay * param
            
            # Update velocity
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + (1 - self.dampening) * param.grad
                
                if self.nesterov:
                    param = param - self.lr * (param.grad + self.momentum * self.velocities[i])
                else:
                    param = param - self.lr * self.velocities[i]
            else:
                param = param - self.lr * param.grad
        
        self.step_count += 1

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.first_moments = []
        self.second_moments = []
        for param in parameters:
            self.first_moments.append(Tensor.zeros_like(param))
            self.second_moments.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one Adam step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad = param.grad + self.weight_decay * param
            
            # Update biased first moment estimate
            self.first_moments[i] = self.beta1 * self.first_moments[i] + (1 - self.beta1) * param.grad
            
            # Update biased second moment estimate
            self.second_moments[i] = self.beta2 * self.second_moments[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Bias correction
            first_moment_corrected = self.first_moments[i] / (1 - self.beta1 ** (self.step_count + 1))
            second_moment_corrected = self.second_moments[i] / (1 - self.beta2 ** (self.step_count + 1))
            
            # Update parameter
            param = param - self.lr * first_moment_corrected / (second_moment_corrected.sqrt() + self.eps)
        
        self.step_count += 1

class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.first_moments = []
        self.second_moments = []
        for param in parameters:
            self.first_moments.append(Tensor.zeros_like(param))
            self.second_moments.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one AdamW step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Update biased first moment estimate
            self.first_moments[i] = self.beta1 * self.first_moments[i] + (1 - self.beta1) * param.grad
            
            # Update biased second moment estimate
            self.second_moments[i] = self.beta2 * self.second_moments[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Bias correction
            first_moment_corrected = self.first_moments[i] / (1 - self.beta1 ** (self.step_count + 1))
            second_moment_corrected = self.second_moments[i] / (1 - self.beta2 ** (self.step_count + 1))
            
            # Update parameter (with decoupled weight decay)
            param = param - self.lr * (first_moment_corrected / (second_moment_corrected.sqrt() + self.eps) + 
                                     self.weight_decay * param)
        
        self.step_count += 1

class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        
        # Initialize squared gradients and velocities
        self.squared_gradients = []
        self.velocities = []
        for param in parameters:
            self.squared_gradients.append(Tensor.zeros_like(param))
            if momentum != 0:
                self.velocities.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one RMSprop step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad = param.grad + self.weight_decay * param
            
            # Update squared gradients
            self.squared_gradients[i] = self.alpha * self.squared_gradients[i] + (1 - self.alpha) * (param.grad ** 2)
            
            # Compute update
            update = param.grad / (self.squared_gradients[i].sqrt() + self.eps)
            
            # Apply momentum if enabled
            if self.momentum != 0:
                self.velocities[i] = self.momentum * self.velocities[i] + self.lr * update
                param = param - self.velocities[i]
            else:
                param = param - self.lr * update
        
        self.step_count += 1

class Adagrad(Optimizer):
    """Adagrad optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01,
                 eps: float = 1e-10, weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize squared gradients
        self.squared_gradients = []
        for param in parameters:
            self.squared_gradients.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one Adagrad step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad = param.grad + self.weight_decay * param
            
            # Update squared gradients
            self.squared_gradients[i] = self.squared_gradients[i] + param.grad ** 2
            
            # Update parameter
            param = param - self.lr * param.grad / (self.squared_gradients[i].sqrt() + self.eps)
        
        self.step_count += 1

class Adadelta(Optimizer):
    """Adadelta optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 1.0,
                 rho: float = 0.9, eps: float = 1e-6, weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize squared gradients and squared updates
        self.squared_gradients = []
        self.squared_updates = []
        for param in parameters:
            self.squared_gradients.append(Tensor.zeros_like(param))
            self.squared_updates.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one Adadelta step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad = param.grad + self.weight_decay * param
            
            # Update squared gradients
            self.squared_gradients[i] = self.rho * self.squared_gradients[i] + (1 - self.rho) * (param.grad ** 2)
            
            # Compute update
            update = param.grad * (self.squared_updates[i].sqrt() + self.eps) / (self.squared_gradients[i].sqrt() + self.eps)
            
            # Update parameter
            param = param - self.lr * update
            
            # Update squared updates
            self.squared_updates[i] = self.rho * self.squared_updates[i] + (1 - self.rho) * (update ** 2)
        
        self.step_count += 1

class Adamax(Optimizer):
    """Adamax optimizer."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.002,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.first_moments = []
        self.second_moments = []
        for param in parameters:
            self.first_moments.append(Tensor.zeros_like(param))
            self.second_moments.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one Adamax step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad = param.grad + self.weight_decay * param
            
            # Update biased first moment estimate
            self.first_moments[i] = self.beta1 * self.first_moments[i] + (1 - self.beta1) * param.grad
            
            # Update biased second moment estimate (using infinity norm)
            self.second_moments[i] = self.beta2 * self.second_moments[i].maximum(param.grad.abs())
            
            # Bias correction
            first_moment_corrected = self.first_moments[i] / (1 - self.beta1 ** (self.step_count + 1))
            
            # Update parameter
            param = param - self.lr * first_moment_corrected / (self.second_moments[i] + self.eps)
        
        self.step_count += 1

class Nadam(Optimizer):
    """Nadam optimizer (Nesterov Adam)."""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.002,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moment estimates
        self.first_moments = []
        self.second_moments = []
        for param in parameters:
            self.first_moments.append(Tensor.zeros_like(param))
            self.second_moments.append(Tensor.zeros_like(param))
    
    def step(self):
        """Perform one Nadam step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad = param.grad + self.weight_decay * param
            
            # Update biased first moment estimate
            self.first_moments[i] = self.beta1 * self.first_moments[i] + (1 - self.beta1) * param.grad
            
            # Update biased second moment estimate
            self.second_moments[i] = self.beta2 * self.second_moments[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Bias correction
            first_moment_corrected = self.first_moments[i] / (1 - self.beta1 ** (self.step_count + 1))
            second_moment_corrected = self.second_moments[i] / (1 - self.beta2 ** (self.step_count + 1))
            
            # Nesterov momentum
            nesterov_momentum = self.beta1 * first_moment_corrected + (1 - self.beta1) * param.grad
            
            # Update parameter
            param = param - self.lr * nesterov_momentum / (second_moment_corrected.sqrt() + self.eps)
        
        self.step_count += 1

# Learning rate schedulers
class LearningRateScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.get_lr()
        self.epoch = 0
    
    def step(self):
        """Update learning rate."""
        self.epoch += 1
        new_lr = self.get_lr()
        self.optimizer.set_lr(new_lr)
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        raise NotImplementedError

class StepLR(LearningRateScheduler):
    """Step learning rate scheduler."""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.base_lr * (self.gamma ** (self.epoch // self.step_size))

class ExponentialLR(LearningRateScheduler):
    """Exponential learning rate scheduler."""
    
    def __init__(self, optimizer: Optimizer, gamma: float):
        super().__init__(optimizer)
        self.gamma = gamma
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.base_lr * (self.gamma ** self.epoch)

class CosineAnnealingLR(LearningRateScheduler):
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.epoch / self.T_max)) / 2

# Utility functions
def create_optimizer(name: str, parameters: List[Tensor], **kwargs) -> Optimizer:
    """Create optimizer by name."""
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizers[name](parameters, **kwargs)

def get_available_optimizers() -> List[str]:
    """Get list of available optimizers."""
    return ['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']

def create_scheduler(name: str, optimizer: Optimizer, **kwargs) -> LearningRateScheduler:
    """Create learning rate scheduler by name."""
    schedulers = {
        'step': StepLR,
        'exponential': ExponentialLR,
        'cosine_annealing': CosineAnnealingLR,
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")
    
    return schedulers[name](optimizer, **kwargs)

def get_available_schedulers() -> List[str]:
    """Get list of available schedulers."""
    return ['step', 'exponential', 'cosine_annealing']
