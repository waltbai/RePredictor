"""Sequence modeling layers."""
import math

import torch
from torch import nn, FloatTensor, BoolTensor
from torch.autograd import Variable


__all__ = [
    "Transformer",
    "PositionEncoder"
]


class PositionEncoder(nn.Module):
    """Position encoder used by transformer."""

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        """Construction method for position encoder.

        Args:
            d_model (int): input dimension.
            seq_len (int): sequence length.
            dropout (float): dropout rate.
                Defaults to 0.1
        """
        super(PositionEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0., seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: FloatTensor):
        """Add position embedding to input tensor.

        Args:
            x (FloatTensor): input, size(batch_size, seq_len, d_model)
        """
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class Transformer(nn.Module):
    """Transformer layer with position encoding."""

    def __init__(self, d_model: int, seq_len: int = 9, nhead: int = 16,
                 dim_feedforward: int = 512, num_layers: int = 1,
                 dropout: float = 0.1):
        """Construction method for transformer.

        Args:
            d_model (int): input and output dimension.
            seq_len (int): sequence length.
                Defaults to 9.
            nhead (int): number of heads.
                Defaults to 16.
            dim_feedforward (int): feedforward dimension.
                Defaults to 512.
            num_layers (int): number of layers.
                Defaults to 1.
            dropout (float): dropout rate.
                Defaults to 0.1.
        """
        super(Transformer, self).__init__()
        self.position_encoder = PositionEncoder(d_model, seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers)

    def forward(self, x: FloatTensor, mask: BoolTensor = None):
        """Forward transformer layer.

        Args:
            x (FloatTensor): input, size(batch_size, seq_len, d_model).
            mask (BoolTensor): mask for input, size(batch_size, seq_len).

        Returns:
            FloatTensor: size(batch_size, seq_len, d_model)
        """
        x = self.position_encoder(x)
        __input = x
        __output = self.transformer(__input, src_key_padding_mask=mask)
        return __output
