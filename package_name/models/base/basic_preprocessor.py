"""Basic preprocessor."""

from abc import ABC, abstractmethod
import os


class BasicPreprocessor(ABC):
    """Basic preprocessor.

    In many cases, same type of models share the same preprocess
    steps (only with a few differences). Therefore, they can inherit
    the same preprocessor to reduce redundant codes. Since different
    task requires different preprocess steps, the common preprocess
    steps can be defined according to your own task.

    Notice: same to BasicPredictor, in ``__init__`` method of subclasses,
    ``super`` method should be called IN THE LAST.
    """

    def __init__(self, config: dict):
        """Construction method for BasicPreprocessor class.

        Args:
            config (dict): configuration.
        """
        self._config = config
        # Source data directory
        self._data_dir = config["data_dir"]
        # Work directory
        self._work_dir = config["work_dir"]
        # Whether to use progress_bar in this model
        self._progress_bar = config["progress_bar"]
        # Whether to overwrite existing files
        self._overwrite = config["overwrite"]
        # Model type: In most cases, same model type shares same preprocessor
        self._model_type = config["model"]["type"]
        # Preprocess data dir
        self._preprocess_dir = os.path.join(
            self._work_dir, f"{self._model_type}_data")
        # Logger
        self._logger = None
        # Build preprocessor
        self.build_preprocessor()

    @abstractmethod
    def build_preprocessor(self):
        """Build the preprocessor."""

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, checkpoint_dir: str):
        """Load preprocessor from the checkpoint.

        Args:
            checkpoint_dir (str): directory of the checkpoint.
        """
