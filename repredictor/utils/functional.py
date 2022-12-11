"""Common functions."""
from typing import Dict


def lookup(word: str,
           dictionary: Dict[str, int],
           default: str = "None") -> int:
    """Look up word in the dictionary for index.

    Args:
        word (str): word
        dictionary (dict[str, int]): dictionary.
        default (str): default word.

    Returns:
        int: the word index.
    """
    return dictionary[word] if word in dictionary else dictionary[default]


def get_value(config: dict, key: str, default=None):
    """Get value from configuration.

    Args:
        config (dict): configuration.
        key (str): key.
        default: default value.
            Defaults to None.
    """
    if key in config:
        return config[key]
    else:
        return default
