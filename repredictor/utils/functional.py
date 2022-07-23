"""Common functions."""


def lookup(word: str,
           dictionary: dict[str, int],
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


def get_value(config: dict, key: str):
    """Get value from configuration.

    Args:
        config (dict): configuration.
        key (str): key.
    """
    if key in config:
        return config[key]
    else:
        return None
