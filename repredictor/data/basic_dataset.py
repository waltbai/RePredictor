"""Basic dataset."""

import math
import pickle
import random
from abc import ABC, abstractmethod

__all__ = ["BasicDataset", "BasicDataLoader"]


class BasicDataset(ABC):
    """Abstract class for dataset."""

    def __init__(self, data):
        """Construction method for BasicDataset.

        Args:
            data: data
        """
        self._data = data

    @abstractmethod
    def __getitem__(self, idx: int):
        """Get an item by index.

        Args:
            idx (int): index
        """

    def __len__(self):
        """Size of dataset."""
        return len(self._data)

    @classmethod
    def from_file(cls, fp: str, **kwargs):
        """Load data from file.

        Args:
            fp (str): file path.
        """
        with open(fp, "rb") as f:
            data = pickle.load(f)
        return cls(data)


class BasicDataLoader(ABC):
    """Abstract class for dataloader."""

    def __init__(self,
                 dataset: BasicDataset,
                 batch_size: int = 1,
                 shuffle: bool = False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        """Iterate batches."""
        num_samples = len(self._dataset)
        batch_size = self._batch_size
        indices = list(range(num_samples))
        if self._shuffle:
            random.shuffle(indices)
        num_batches = math.ceil(num_samples / batch_size)
        for i in range(num_batches):
            batch_indices = indices[batch_size * i:batch_size * (i + 1)]
            yield self.construct_batch(batch_indices)

    @abstractmethod
    def construct_batch(self, indices: list):
        """Construct a batch via indices given.

        Args:
            indices (list): indices in the batch.
        """
