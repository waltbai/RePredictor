"""Configurations."""

import yaml

__all__ = ["load_config", "save_config"]


def load_config(config_path="config/scpredictor.yml"):
    """Load configuration from file.

    Args:
        config_path (str, optional): config file path.
            Defaults to "config/scpredictor.yml".

    Returns:
        dict: configuration object
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def save_config(config, config_path):
    """Save configuration to file.

    Args:
        config (dict): configuration file.
        config_path (str): configuration file path.
    """
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
