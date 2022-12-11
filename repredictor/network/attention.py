"""Attention layers."""
import math

import torch
from torch import nn, FloatTensor, BoolTensor


__all__ = [
    "AdditiveAttention",
    "DotAttention",
    "ScaledDotAttention",
    "AverageAttention",
    "construct_attention_layer",
]


class AdditiveAttention(nn.Module):
    """Additive attention function."""

    def __init__(self,
                 event_dim: int,
                 directions: int = 1):
        """Construction method for additive attention.

        Args:
            event_dim:
            directions:
        """
        super(AdditiveAttention, self).__init__()
        self.ffn = nn.Linear(event_dim * directions * 2, 1)

    def forward(self,
                context: FloatTensor,
                choices: FloatTensor,
                mask: BoolTensor = None):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim).
            choices (FloatTenosr): choices events,
                size(batch_size, *, 1 or seq_len, event_dim).
            mask (BoolTensor): event mask,
                size(batch_size, *, seq_len).

        Returns:
            FloatTensor: attention weight,
                size(batch_size, *, seq_len)
        """
        if choices.size(-2) == 1:
            choices = choices.expand(context.size())
        __input = torch.cat([context, choices], dim=-1)
        weight = self.ffn(__input).sequeeze()
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


class DotAttention(nn.Module):
    """Dot attention function."""

    def __init__(self):
        """Construction method for dot attention."""
        super(DotAttention, self).__init__()

    def forward(self,
                context: FloatTensor,
                choices: FloatTensor,
                mask: BoolTensor = None):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim).
            choices (FloatTenosr): choices events,
                size(batch_size, *, 1 or seq_len, event_dim).
            mask (BoolTensor): event mask,
                size(batch_size, *, seq_len).

        Returns:
            FloatTensor: attention weight,
                size(batch_size, *, seq_len)
        """
        weight = (context * choices).sum(-1)
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


class ScaledDotAttention(nn.Module):
    """Scaled dot attention function."""

    def __init__(self):
        """Construction method for scaled-dot attention."""
        super(ScaledDotAttention, self).__init__()

    def forward(self,
                context: FloatTensor,
                choices: FloatTensor,
                mask: BoolTensor = None):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim).
            choices (FloatTenosr): choices events,
                size(batch_size, *, 1 or seq_len, event_dim).
            mask (BoolTensor): event mask,
                size(batch_size, *, seq_len).

        Returns:
            FloatTensor: attention weight,
                size(batch_size, *, seq_len)
        """
        event_dim = context.size(-1)
        weight = (context * choices).sum(-1) / math.sqrt(event_dim)
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


class AverageAttention(nn.Module):
    """Average attention function."""

    def __init__(self):
        """Construction method for average attention."""
        super(AverageAttention, self).__init__()

    def forward(self,
                context: FloatTensor,
                choices: FloatTensor,
                mask: BoolTensor = None):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, *, seq_len, event_dim).
            choices (FloatTenosr): choices events,
                size(batch_size, *, 1 or seq_len, event_dim).
            mask (BoolTensor): event mask,
                size(batch_size, *, seq_len).

        Returns:
            FloatTensor: attention weight,
                size(batch_size, *, seq_len)
        """
        weight = context.new_ones(context.size()[:-1], dtype=torch.float)
        if mask is not None:
            weight = weight.masked_fill(mask, -1e9)
        attn = torch.softmax(weight, dim=-1)
        return attn


def construct_attention_layer(func_name: str,
                              event_dim: int = None,
                              directions: int = None):
    """Construct attention layer according to function name.

    Args:
        func_name (str): the name of attention layer.
            "additive", "dot", "scaled-dot" or "average".
        event_dim (int): in case needed.
        directions (int): in case needed.
    Raises:
        KeyError
    """
    if func_name == "additive":
        return AdditiveAttention(event_dim=event_dim,
                                 directions=directions)
    elif func_name == "dot":
        return DotAttention()
    elif func_name == "scaled-dot":
        return ScaledDotAttention()
    elif func_name == "average":
        return AverageAttention()
    else:
        raise KeyError(f"Unknown attention function '{func_name}'!")
