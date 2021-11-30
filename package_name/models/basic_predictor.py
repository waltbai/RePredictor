"""Basic predictor."""
import os
from abc import ABC, abstractmethod

import torch

from package_name.utils.config import load_config, save_config


class BasicPredictor(ABC):
    """Basic predictor class."""

    def __init__(self, config, device=0):
        self._config = config
        self._data_dir = config["data_dir"]
        self._work_dir = config["work_dir"]
        # Whether to use progress_bar in this model
        self._progress_bar = config["progress_bar"]
        self._model_name = config["model"]["name"]
        self._model_dir = os.path.join(
            self._work_dir, "models", self._model_name)
        self._device = device
        # Specified by each predictor
        self._model = None
        self._optimizer = None
        self._logger = None
        self.build_model()

    @abstractmethod
    def build_model(self):
        """Initialize a model."""

    @abstractmethod
    def train(self):
        """Train on model."""

    @abstractmethod
    def predict(self):
        """Predict on model."""

    @abstractmethod
    def evaluate(self):
        """Evaluate on model."""

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, device=0, verbose=True):
        """Load model from checkpoint."""
        config_path = os.path.join(checkpoint_dir, "config.yml")
        config = load_config(config_path)
        predictor = BasicPredictor(config, device)
        if verbose:
            predictor._logger.info(
                f"Loading checkpoint from {checkpoint_dir}.")
        model_path = os.path.join(checkpoint_dir, "model.pt")
        predictor.load_model_params(model_path, verbose)
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        predictor.load_optimizer_params(optimizer_path, verbose)
        return predictor

    def save_checkpoint(self, checkpoint, verbose=True):
        """Save a checkpoint, either use 'epoch-N' or 'best'."""
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

    def save_model_params(self, path, verbose=True):
        """Save model parameters."""
        if os.path.exists(path):
            if verbose:
                self._logger.warning(f"Overwrite model parameters {path}!")
        else:
            if verbose:
                self._logger.info(f"Save model parameters to {path}.")
        torch.save(self._model.state_dict(), path)

    def load_model_params(self, path, verbose=True):
        """Load model parameters."""
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

    def save_optimizer_params(self, path, verbose=True):
        """Save optimizer parameters."""
        if os.path.exists(path):
            if verbose:
                self._logger.warning(f"Overwrite optimizer parameters {path}!")
        else:
            if verbose:
                self._logger.info(f"Save optimizer parameters to {path}.")
        torch.save(self._optimizer.state_dict(), path)

    def load_optimizer_params(self, path, verbose=True):
        """Load optimizer parameters."""
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
