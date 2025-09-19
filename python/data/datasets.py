"""
Dataset classes for TensorCore

This module provides dataset classes for loading and managing data.
"""

from ..tensorcore_core import Tensor
from typing import List, Tuple, Optional, Callable, Union
import random

class Dataset:
    """Base class for all datasets."""
    
    def __init__(self):
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Return a sample and its label."""
        raise NotImplementedError

class TensorDataset(Dataset):
    """Dataset for tensors."""
    
    def __init__(self, data: Tensor, labels: Tensor, transform: Optional[Callable] = None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
        if data.shape[0] != labels.shape[0]:
            raise ValueError("Data and labels must have the same number of samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Return a sample and its label."""
        sample = self.data[index]
        label = self.labels[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label

class DataLoader:
    """Data loader for batching and shuffling data."""
    
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False,
                 sampler: Optional['Sampler'] = None, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.drop_last = drop_last
        
        if sampler is None:
            if shuffle:
                self.sampler = RandomSampler(dataset)
            else:
                self.sampler = SequentialSampler(dataset)
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Return iterator over batches."""
        return DataLoaderIterator(self)

class DataLoaderIterator:
    """Iterator for DataLoader."""
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.sampler_iter = iter(dataloader.sampler)
        self.batch_size = dataloader.batch_size
        self.drop_last = dataloader.drop_last
    
    def __next__(self) -> Tuple[Tensor, Tensor]:
        """Return next batch."""
        batch_data = []
        batch_labels = []
        
        for _ in range(self.batch_size):
            try:
                index = next(self.sampler_iter)
                sample, label = self.dataloader.dataset[index]
                batch_data.append(sample)
                batch_labels.append(label)
            except StopIteration:
                if len(batch_data) == 0:
                    raise StopIteration
                if self.drop_last:
                    raise StopIteration
                break
        
        if len(batch_data) == 0:
            raise StopIteration
        
        # Stack tensors
        batch_data_tensor = Tensor.stack(batch_data)
        batch_labels_tensor = Tensor.stack(batch_labels)
        
        return batch_data_tensor, batch_labels_tensor

class Sampler:
    """Base class for samplers."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def __iter__(self):
        """Return iterator over indices."""
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.dataset)

class RandomSampler(Sampler):
    """Random sampler."""
    
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """Return iterator over shuffled indices."""
        random.shuffle(self.indices)
        return iter(self.indices)

class SequentialSampler(Sampler):
    """Sequential sampler."""
    
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """Return iterator over sequential indices."""
        return iter(self.indices)

class WeightedRandomSampler(Sampler):
    """Weighted random sampler."""
    
    def __init__(self, dataset: Dataset, weights: List[float], num_samples: int):
        super().__init__(dataset)
        self.weights = weights
        self.num_samples = num_samples
        
        if len(weights) != len(dataset):
            raise ValueError("Number of weights must match dataset size")
    
    def __iter__(self):
        """Return iterator over weighted random indices."""
        indices = []
        for _ in range(self.num_samples):
            index = random.choices(range(len(self.dataset)), weights=self.weights)[0]
            indices.append(index)
        return iter(indices)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

class SubsetRandomSampler(Sampler):
    """Subset random sampler."""
    
    def __init__(self, dataset: Dataset, indices: List[int]):
        super().__init__(dataset)
        self.indices = indices
    
    def __iter__(self):
        """Return iterator over shuffled subset indices."""
        random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.indices)

class BatchSampler(Sampler):
    """Batch sampler."""
    
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        super().__init__(sampler.dataset)
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    
    def __iter__(self):
        """Return iterator over batches of indices."""
        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

# Utility functions
def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """Collate function for batching."""
    data, labels = zip(*batch)
    return Tensor.stack(data), Tensor.stack(labels)

def default_collate(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    """Default collate function."""
    return collate_fn(batch)
