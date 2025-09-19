"""
Data utility functions for TensorCore

This module provides various data preprocessing and loading utilities.
"""

from ..tensorcore_core import Tensor
import random
import math
from typing import List, Tuple, Optional, Union
import os

class RandomGenerator:
    """Random number generator for TensorCore."""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or random.randint(0, 2**32 - 1)
        random.seed(self.seed)
    
    def set_seed(self, seed: int):
        """Set random seed."""
        self.seed = seed
        random.seed(seed)
    
    def uniform(self, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Generate uniform random number."""
        return random.uniform(min_val, max_val)
    
    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Generate normal random number."""
        return random.gauss(mean, std)
    
    def uniform_int(self, min_val: int, max_val: int) -> int:
        """Generate uniform random integer."""
        return random.randint(min_val, max_val)
    
    def bernoulli(self, p: float = 0.5) -> bool:
        """Generate Bernoulli random variable."""
        return random.random() < p
    
    def choice(self, n: int, k: int, replace: bool = True) -> List[int]:
        """Random choice without replacement."""
        if replace:
            return [random.randint(0, n-1) for _ in range(k)]
        else:
            return random.sample(range(n), min(k, n))
    
    def permutation(self, n: int) -> List[int]:
        """Random permutation of integers 0 to n-1."""
        return random.sample(range(n), n)
    
    def shuffle(self, data: List) -> List:
        """Shuffle a list in place."""
        random.shuffle(data)
        return data

# Global random generator
_rng = RandomGenerator()

def set_random_seed(seed: int):
    """Set global random seed."""
    global _rng
    _rng.set_seed(seed)

class DataPreprocessor:
    """Data preprocessing utilities."""
    
    @staticmethod
    def normalize(data: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
        """Normalize data to have specified mean and std."""
        data_mean = data.mean()
        data_std = data.std()
        return (data - data_mean) / data_std * std + mean
    
    @staticmethod
    def min_max_scale(data: Tensor, min_val: float = 0.0, max_val: float = 1.0) -> Tensor:
        """Scale data to specified range."""
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
    
    @staticmethod
    def robust_scale(data: Tensor) -> Tensor:
        """Robust scaling using median and IQR."""
        data_median = data.median()
        q75 = data.percentile(75)
        q25 = data.percentile(25)
        iqr = q75 - q25
        return (data - data_median) / iqr
    
    @staticmethod
    def unit_scale(data: Tensor) -> Tensor:
        """Scale data to unit norm."""
        return data / data.norm()
    
    @staticmethod
    def standardize(data: Tensor) -> Tuple[Tensor, Tuple[float, float]]:
        """Standardize data and return scaling parameters."""
        data_mean = data.mean()
        data_std = data.std()
        standardized = (data - data_mean) / data_std
        return standardized, (data_mean, data_std)
    
    @staticmethod
    def inverse_standardize(data: Tensor, mean: float, std: float) -> Tensor:
        """Inverse standardize data using scaling parameters."""
        return data * std + mean
    
    @staticmethod
    def one_hot_encode(labels: Tensor, num_classes: int) -> Tensor:
        """One-hot encode labels."""
        batch_size = labels.size
        one_hot = Tensor.zeros((batch_size, num_classes))
        for i in range(batch_size):
            one_hot[i, int(labels[i])] = 1.0
        return one_hot
    
    @staticmethod
    def one_hot_decode(one_hot: Tensor) -> Tensor:
        """Decode one-hot encoded labels."""
        return one_hot.argmax(axis=1)
    
    @staticmethod
    def train_test_split(X: Tensor, y: Tensor, test_size: float = 0.2, 
                        random_state: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split data into train and test sets."""
        if random_state is not None:
            set_random_seed(random_state)
        
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test
        
        # Random permutation of indices
        indices = _rng.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split data
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def k_fold_split(X: Tensor, y: Tensor, k: int = 5, 
                    random_state: Optional[int] = None) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """K-fold cross-validation split."""
        if random_state is not None:
            set_random_seed(random_state)
        
        n_samples = X.shape[0]
        fold_size = n_samples // k
        indices = _rng.permutation(n_samples)
        
        folds = []
        for i in range(k):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k - 1 else n_samples
            
            test_indices = indices[start_idx:end_idx]
            train_indices = [idx for idx in indices if idx not in test_indices]
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            folds.append((X_train, X_test, y_train, y_test))
        
        return folds
    
    @staticmethod
    def add_noise(data: Tensor, noise_level: float = 0.1) -> Tensor:
        """Add Gaussian noise to data."""
        noise = Tensor.random_normal(data.shape, 0.0, noise_level)
        return data + noise
    
    @staticmethod
    def random_rotation(data: Tensor, max_angle: float = 15.0) -> Tensor:
        """Apply random rotation to data (placeholder implementation)."""
        # This would be implemented for image data
        return data
    
    @staticmethod
    def random_shift(data: Tensor, max_shift: float = 0.1) -> Tensor:
        """Apply random shift to data (placeholder implementation)."""
        # This would be implemented for image data
        return data
    
    @staticmethod
    def random_flip(data: Tensor, flip_prob: float = 0.5) -> Tensor:
        """Apply random flip to data (placeholder implementation)."""
        # This would be implemented for image data
        return data

class DataLoader:
    """Data loading utilities."""
    
    @staticmethod
    def load_csv(filename: str, has_header: bool = True, 
                delimiter: str = ',') -> Tuple[Tensor, Tensor]:
        """Load data from CSV file."""
        import csv
        
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            if has_header:
                next(reader)  # Skip header
            
            for row in reader:
                data.append([float(x) for x in row])
        
        # Convert to tensor
        data_tensor = Tensor(data)
        
        # Split features and labels (assuming last column is label)
        X = data_tensor[:, :-1]
        y = data_tensor[:, -1]
        
        return X, y
    
    @staticmethod
    def load_image(filename: str) -> Tensor:
        """Load image from file (placeholder implementation)."""
        # This would be implemented using PIL or OpenCV
        raise NotImplementedError("Image loading not yet implemented")
    
    @staticmethod
    def save_image(image: Tensor, filename: str):
        """Save image to file (placeholder implementation)."""
        # This would be implemented using PIL or OpenCV
        raise NotImplementedError("Image saving not yet implemented")
    
    @staticmethod
    def load_mnist(data_dir: str) -> Tuple[Tensor, Tensor]:
        """Load MNIST dataset (placeholder implementation)."""
        # This would be implemented to load MNIST data
        raise NotImplementedError("MNIST loading not yet implemented")
    
    @staticmethod
    def load_cifar10(data_dir: str) -> Tuple[Tensor, Tensor]:
        """Load CIFAR-10 dataset (placeholder implementation)."""
        # This would be implemented to load CIFAR-10 data
        raise NotImplementedError("CIFAR-10 loading not yet implemented")
    
    @staticmethod
    def load_text_lines(filename: str) -> List[str]:
        """Load text file as list of lines."""
        with open(filename, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file]
    
    @staticmethod
    def load_text(filename: str) -> str:
        """Load text file as single string."""
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()

# Global utility functions
def create_identity_matrix(size: int) -> Tensor:
    """Create identity matrix."""
    return Tensor.eye(size)

def create_zeros(shape: Tuple[int, ...]) -> Tensor:
    """Create tensor of zeros."""
    return Tensor.zeros(shape)

def create_ones(shape: Tuple[int, ...]) -> Tensor:
    """Create tensor of ones."""
    return Tensor.ones(shape)

def create_range(start: float, stop: float, step: float = 1.0) -> Tensor:
    """Create tensor with range of values."""
    return Tensor.arange(start, stop, step)

def print_tensor_info(tensor: Tensor, name: str = ""):
    """Print tensor information."""
    print(f"{name}: shape={tensor.shape}, dtype=float, size={tensor.size}")
    print(f"  min={tensor.min()}, max={tensor.max()}, mean={tensor.mean()}, std={tensor.std()}")

def print_memory_usage():
    """Print memory usage information."""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def print_configuration():
    """Print TensorCore configuration."""
    print("TensorCore Configuration:")
    print(f"  Random seed: {_rng.seed}")
    print(f"  Available functions: {len(dir(Tensor))}")
