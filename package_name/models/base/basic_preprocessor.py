"""Basic preprocessor."""

import os
from abc import ABC, abstractmethod

__all__ = ["BasicPreprocessor"]


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
    def load(cls, load_dir: str):
        """Load preprocessor from the checkpoint.

        Args:
            load_dir (str): directory to load.
        """
    
    @abstractmethod
    def save(self, save_dir: str):
        """Save checkpoint.

        Notice: this method will NOT save config file, since the model
        should save it. If preprocessor contains files like word dictionary,
        save it.

        Args:
            save_dir (str): checkpoint name.
        """
