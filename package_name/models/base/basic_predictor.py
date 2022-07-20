"""Basic predictor."""

import os
from abc import ABC, abstractmethod

import torch
from package_name.models.base.basic_dataset import BasicDataset
from package_name.utils.config import load_config, save_config

__all__ =["BasicPredictor"]


class BasicPredictor(ABC):
    """Basic predictor class.

    Notice: In the :meth:`__init__` method of subclasses,
    :meth:`super` method should be called IN THE LAST.
    """

    def __init__(self,
                 config: dict,
                 device: str = "cpu"):
        """Construction method of BasicPredictor class.

        Args:
            config (dict): configuration.
            device (str, optional): device name.
                Defaults to "cpu".
        """
        self._config = config
        # Source data directory
        self._data_dir = config["data_dir"]
        # Work directory
        self._work_dir = config["work_dir"]
        # Whether to use progress_bar in this model
        self._progress_bar = config["progress_bar"]
        # Model type which usually correspond to a neural model
        self._model_type = config["model"]["type"]
        # Model name which should be globally unique
        self._model_name = config["model"]["name"]
        # Set model directory for checkpoints
        if "model_dir" in config:
            base_model_dir = config["model_dir"]
        else:
            base_model_dir = os.path.join(self._work_dir, "models")
        self._model_dir = os.path.join(base_model_dir, self._model_name)
        # Set device for the model
        self._device = device
        # ===== Specified by each predictor =====
        # The neural model
        self._model = None
        # The optimizer
        self._optimizer = None
        # The logger
        self._logger = None
        # Build model
        self.build_model()

    @abstractmethod
    def build_model(self):
        """Initialize a model according to the configuration.

        Model, optimizer and logger should be initialized here.
        """

    @abstractmethod
    def train(self,
              train_set: BasicDataset = None,
              train_fp: str = None,
              verbose: bool = True) -> None:
        """Train the model on train set.

        It is suggested to save the checkpoint to self._model_dir.

        Args:
            train_set (BasicDataset, optional): train set.
                Defaults to None.
            train_fp (str, optional): train file path.
                In many cases, train set is too large to fully load into memory.
                Then, this argument should be a directory.
                Only used when train_set is None.
                Defaults to None.
            verbose (bool): whether to print training details in logger.
                Defaults to True.
        """

    @abstractmethod
    def validate(self,
                 valid_set: BasicDataset = None,
                 valid_fp: str = None,
                 verbose: bool = False) -> dict or float:
        """Evaluate the model on dev/valid set.

        Args:
            valid_set (BasicDataset, optional): dev/valid set.
                Defaults to None.
            valid_fp (str, optional): the dev/valid file path.
                Only used when valid_set is None.
                Defaults to None.
            verbose (bool): whether to print evaluation details in logger.
                Defaults to False.

        Returns:
            float or dict: performance under evaluation metric.
        """

    @abstractmethod
    def test(self,
             test_set: BasicDataset = None,
             test_fp: str = None,
             verbose: bool = True) -> dict or float:
        """Evaluate the model on test set.

        Args:
            test_set (dataset, optional): test set.
                Defaults to None.
            test_fp (str, optional): test file path.
                Only used when test_set is None.
                Defaults to None.
            verbose (bool): whether to print evaluation details in logger.
                Defaults to True.

        Returns:
            float or dict: performance under evaluation metric.
        """

    @abstractmethod
    def predict(self,
                pred_set: BasicDataset = None,
                pred_fp: str = None):
        """Predict samples.

        Args:
            pred_set (dataset, optional): dataset to be predict.
                Defaults to None.
            pred_fp (str, optional): dataset file path.
                Only used when pred_set is None.
                Defaults to None.

        Returns:
            Any: results.
        """

    @abstractmethod
    def evaluate(self, pred_label, true_label) -> dict or float:
        """Evaluate the prediction results.

        Args:
            pred_label: predict labels
            true_label: true labels

        Returns:
            float or dict: performance under evaluation metric.
        """

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_dir: str,
                        device: str = "cpu",
                        verbose: bool = True):
        """Load model from checkpoint.

        Args:
            checkpoint_dir (str): checkpoint directory
            device (str, optional): the computing device.
                Defaults to "cpu".
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.

        Returns:
            BasicPredictor: the trained model
        """
        config_path = os.path.join(checkpoint_dir, "config.yml")
        config = load_config(config_path)
        predictor = cls(config, device)
        if verbose:
            predictor._logger.info(
                f"Loading checkpoint from {checkpoint_dir}.")
        model_path = os.path.join(checkpoint_dir, "model.pt")
        predictor.load_model_params(model_path, verbose)
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        predictor.load_optimizer_params(optimizer_path, verbose)
        return predictor

    def save_checkpoint(self,
                        checkpoint: str,
                        verbose: bool = True) -> None:
        """Save checkpoint.

        Args:
            checkpoint (str): unique name to specify a checkpoint.
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.
        """
        checkpoint_dir = os.path.join(self._model_dir, checkpoint)
        # Create directory for checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save config file
        config_path = os.path.join(checkpoint_dir, "config.yml")
        save_config(self._config, config_path)
        # Save model parameters
        model_path = os.path.join(checkpoint_dir, "model.pt")
        self.save_model_params(model_path, verbose)
        # Save optimizer parameters
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        self.save_optimizer_params(optimizer_path, verbose)
        # logging
        if verbose:
            self._logger.info(
                f"Save checkpoint {checkpoint} to {checkpoint_dir}.")

    def save_model_params(self,
                          path: str,
                          verbose: bool = True) -> None:
        """Save model parameters.

        Args:
            path (str): model file path
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.
        """
        if os.path.exists(path):
            if verbose:
                self._logger.warning(f"Overwrite model parameters {path}!")
        else:
            if verbose:
                self._logger.info(f"Save model parameters to {path}.")
        # Defaults save to cpu device
        torch.save(self._model.cpu().state_dict(), path)

    def load_model_params(self,
                          path: str,
                          verbose: bool = True) -> None:
        """Load model parameters.

        Args:
            path (str): model file path
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.
        """
        if not os.path.exists(path):
            if verbose:
                self._logger.warning(
                    f"Fail to load model parameters from {path}! "
                    f"Use initial parameters.")
        else:
            if verbose:
                self._logger.info(f"Load model parameters from {path}.")
            model_params = torch.load(path, map_location=self._device)
            self._model.load_state_dict(model_params)

    def save_optimizer_params(self,
                              path: str,
                              verbose: bool = True) -> None:
        """Save optimizer parameters.

        Args:
            path (str): optimizer file path.
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.
        """
        if os.path.exists(path):
            if verbose:
                self._logger.warning(f"Overwrite optimizer parameters {path}!")
        else:
            if verbose:
                self._logger.info(f"Save optimizer parameters to {path}.")
        torch.save(self._optimizer.state_dict(), path)

    def load_optimizer_params(self,
                              path: str,
                              verbose: bool = True) -> None:
        """Load optimizer parameters.

        Args:
            path (str): optimizer file path.
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.
        """
        if not os.path.exists(path):
            if verbose:
                self._logger.warning(
                    f"Fail to load optimizer parameters from {path}! "
                    f"Use initial optimizer parameters.")
        else:
            if verbose:
                self._logger.info(f"Load optimizer parameters from {path}.")
            optimizer_params = torch.load(path, map_location=self._device)
            self._optimizer.load_state_dict(optimizer_params)

    def to(self, device: str):
        """Switch model to a new device.

        Notice: though this method follows the style of torch,
            it DOES NOT create a new model object!

        Args:
            device (str): New device name.
        """
        # Set device
        self._device = device
        # Switch neural model
        if self._model is not None:
            self._model = self._model.to(device)
        # Switch optimizer
        if self._optimizer is not None:
            self._optimizer = self._optimizer.to(device)
        return self
