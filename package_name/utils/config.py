"""Load configurations."""

import yaml


def load_config(config_path="config/default.yml"):
    """Get config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config


def save_config(config, config_path):
    """Save config file."""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
