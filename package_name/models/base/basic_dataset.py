"""Basic dataset."""

from abc import ABC, abstractmethod


class BasicDataset(ABC):
    """Basic dataset."""

    def __init__(self,
                 data,
                 train: bool = False):
        self._data = data
        self._train = train

    def train(self, mode=True) -> None:
        """Set training mode.

        Args:
            mode (bool): Whether set to train mode.
                Defaults: True
        """
        self._train = mode

    def predict(self) -> None:
        """Set to predict mode."""
        self._train = False

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
