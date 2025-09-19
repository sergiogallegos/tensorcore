"""
TensorCore - Educational Machine Learning Library

A C++ machine learning library designed for educational purposes to understand
the core mathematics and implementations behind popular ML libraries.
"""

__version__ = "1.0.0"
__author__ = "TensorCore Contributors"
__email__ = "tensorcore@example.com"

# Import the core module
try:
    from .tensorcore_core import *
except ImportError as e:
    raise ImportError(
        "Failed to import TensorCore core module. "
        "Make sure the C++ library is properly built and installed. "
        f"Original error: {e}"
    )

# Import submodules
from . import nn
from . import optim
from . import utils
from . import data

# Make commonly used classes available at the top level
from .tensorcore_core import Tensor, tensor, zeros, ones, eye, arange, linspace
from .tensorcore_core import random_normal, random_uniform

# Version info
__version_info__ = tuple(map(int, __version__.split('.')))

# Module metadata
__all__ = [
    # Core tensor operations
    'Tensor', 'tensor', 'zeros', 'ones', 'eye', 'arange', 'linspace',
    'random_normal', 'random_uniform',
    
    # Mathematical operations
    'add', 'subtract', 'multiply', 'divide', 'power', 'mod',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'log', 'log2', 'log10', 'exp', 'exp2', 'expm1', 'log1p',
    'sqrt', 'cbrt', 'square', 'reciprocal', 'rsqrt',
    'floor', 'ceil', 'round', 'trunc', 'rint',
    'abs', 'fabs', 'sign', 'copysign',
    
    # Comparison operations
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'logical_and', 'logical_or', 'logical_xor', 'logical_not',
    
    # Reduction operations
    'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'prod', 'std', 'var',
    
    # Linear algebra
    'matmul', 'dot', 'outer', 'cross', 'norm', 'transpose', 'conjugate',
    'hermitian', 'trace', 'det', 'inv', 'pinv', 'solve', 'lstsq',
    'eig', 'svd', 'eigh',
    
    # Broadcasting and utility operations
    'broadcast_to', 'concatenate', 'stack', 'split', 'tile', 'repeat', 'pad',
    'histogram', 'percentile', 'quantile', 'median',
    
    # Convolution operations
    'conv1d', 'conv2d', 'max_pool1d', 'max_pool2d',
    
    # Gradient operations
    'gradient', 'hessian',
    
    # Submodules
    'nn', 'optim', 'utils', 'data',
    
    # Version info
    '__version__', '__version_info__', '__author__', '__email__'
]

# Convenience functions for common operations
def create_tensor(data, shape=None, dtype=None, requires_grad=False):
    """Create a tensor from data with optional shape and dtype specification."""
    if shape is None:
        if hasattr(data, '__len__') and not isinstance(data, str):
            shape = (len(data),)
        else:
            shape = (1,)
    
    tensor = Tensor(shape)
    if hasattr(data, '__iter__') and not isinstance(data, str):
        tensor.data = list(data)
    else:
        tensor.fill(data)
    
    if requires_grad:
        tensor.set_requires_grad(True)
    
    return tensor

def from_numpy(array):
    """Convert a NumPy array to a TensorCore tensor."""
    # This would be implemented in the C++ bindings
    raise NotImplementedError("NumPy conversion not yet implemented")

def to_numpy(tensor):
    """Convert a TensorCore tensor to a NumPy array."""
    # This would be implemented in the C++ bindings
    raise NotImplementedError("NumPy conversion not yet implemented")

# Configuration
def set_num_threads(num_threads):
    """Set the number of threads for parallel operations."""
    # This would be implemented in the C++ bindings
    pass

def get_num_threads():
    """Get the current number of threads for parallel operations."""
    # This would be implemented in the C++ bindings
    return 1

def set_seed(seed):
    """Set the random seed for reproducible results."""
    # This would be implemented in the C++ bindings
    pass

# Error handling
class TensorCoreError(Exception):
    """Base exception class for TensorCore errors."""
    pass

class ShapeError(TensorCoreError):
    """Exception raised for tensor shape errors."""
    pass

class DimensionError(TensorCoreError):
    """Exception raised for tensor dimension errors."""
    pass

class ComputationError(TensorCoreError):
    """Exception raised for computation errors."""
    pass

# Module initialization
def _initialize():
    """Initialize the TensorCore module."""
    # Set default configuration
    set_num_threads(1)
    
    # Initialize random number generator
    import random
    set_seed(random.randint(0, 2**32 - 1))

# Initialize the module
_initialize()
