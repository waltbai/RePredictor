"""Basic dataset."""

import math
import random
from abc import ABC, abstractmethod

__all__ = ["BasicDataset", "BasicDataLoader"]


class BasicDataset(ABC):
    """Abstract class for dataset."""

    def __init__(self,
                 data,
                 train: bool = False):
        """Construction method for BasicDataset.

        Args:
            data: data
            train (bool, optional): whether used in train mode.
                Defaults to False.
        """
        self._data = data
        self._train = train

    def train_mode(self, mode=True) -> None:
        """Set training mode.

        Args:
            mode (bool): Whether set to train mode.
                Defaults: True
        """
        self._train = mode

    def predict_mode(self) -> None:
        """Set to predict mode."""
        self._train = False

    def __getitem__(self, idx: int):
        """Get an item by index.

        Args:
            idx (int): index
        """
        if self._train:
            return self.get_input_and_output(idx)
        return self.get_input(idx)

    def __len__(self):
        """Size of dataset."""
        return len(self._data)

    @abstractmethod
    def get_input(self, idx: int):
        """Get input by index.

        This method works in predict mode,
        in which there are no true labels for samples.

        Args:
            idx (int): index of the input.

        Returns:
            Any: input.
        """

    @abstractmethod
    def get_input_and_output(self, idx: int):
        """Get input and output.

        This method works in train mode,
            in which there exists true labels for samples.

        Args:
            idx (int): index of the sample.

        Returns:
            Any: input.
            Any: output.
        """


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
