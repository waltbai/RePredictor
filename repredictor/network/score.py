"""Scoring layers."""
import torch
from torch import nn, FloatTensor


__all__ = [
    "FusionScore",
    "EuclideanScore",
    "ManhattanScore",
    "CosineScore",
    "construct_score_layer",
]


class FusionScore(nn.Module):
    """Fusion score."""

    def __init__(self, event_dim: int, directions: int = 1):
        super(FusionScore, self).__init__()
        self.context_ffn = nn.Linear(event_dim * directions, 1)
        self.choice_ffn = nn.Linear(event_dim * directions, 1)

    def forward(self, context: FloatTensor, choices: FloatTensor):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim)
            choices (FloatTensor): choices events,
                size(batch_size, *, seq_len or 1, event_dim)

        Returns:
            FloatTensor: score, size(batch_size, *, seq_len)
        """
        context_score = self.context_ffn(context).squeeze(-1)
        choices_score = self.choice_ffn(choices).squeeze(-1)
        return context_score + choices_score


class EuclideanScore(nn.Module):
    """Euclidean score."""

    def __init__(self):
        super(EuclideanScore, self).__init__()

    def forward(self, context: FloatTensor, choices: FloatTensor):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim)
            choices (FloatTensor): choices events,
                size(batch_size, *, seq_len or 1, event_dim)

        Returns:
            FloatTensor: score, size(batch_size, *, seq_len)
        """
        return -torch.sqrt(torch.pow(context - choices, 2.).sum(-1))


class ManhattanScore(nn.Module):
    """Manhattan score."""

    def __init__(self):
        super(ManhattanScore, self).__init__()

    def forward(self, context: FloatTensor, choices: FloatTensor):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim)
            choices (FloatTensor): choices events,
                size(batch_size, *, seq_len or 1, event_dim)

        Returns:
            FloatTensor: score, size(batch_size, *, seq_len)
        """
        return -torch.abs(context - choices).sum(-1)


class CosineScore(nn.Module):
    """Cosine score."""

    def __init__(self):
        super(CosineScore, self).__init__()

    def forward(self, context: FloatTensor, choices: FloatTensor):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim)
            choices (FloatTensor): choices events,
                size(batch_size, *, seq_len or 1, event_dim)

        Returns:
            FloatTensor: score, size(batch_size, *, seq_len)
        """
        inner_prod = (context * choices).sum(-1)
        context_length = torch.sqrt(torch.pow(context, 2.).sum(-1))
        choice_length = torch.sqrt(torch.pow(choices, 2.).sum(-1))
        score = inner_prod / context_length / choice_length
        return score


def construct_score_layer(func_name: str,
                          event_dim: int = None,
                          directions: int = None):
    """Construct score layer according to the score function name.

    Args:
        func_name (str): score function name.
            "fusion", "euclidean", "manhattan" or "cosine".
        event_dim (int): in case needed.
        directions (int): in case needed.
    """
    if func_name == "fusion":
        return FusionScore(event_dim=event_dim,
                           directions=directions)
    elif func_name == "euclidean":
        return EuclideanScore()
    elif func_name == "manhattan":
        return ManhattanScore()
    elif func_name == "cosine":
        return CosineScore()
    else:
        raise KeyError(f"Unknown score function '{func_name}'!")
