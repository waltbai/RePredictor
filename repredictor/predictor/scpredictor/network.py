"""Network for SCPredictor."""

import torch
from torch import nn, LongTensor, FloatTensor

from repredictor.network.attention import construct_attention_layer
from repredictor.network.embedding import Embedding
from repredictor.network.event_encoder import EventFusionEncoder
from repredictor.network.score import construct_score_layer
from repredictor.network.sequence_model import Transformer


def append_choices_to_context(context: FloatTensor, choices: FloatTensor):
    """Append choices to the end of context.

    Args:
        context (FloatTensor): context events,
            shape(batch_size, context_size, event_repr_size)
        choices (FloatTensor): choices events,
            shape(batch_size, num_choices, event_repr_size)
    :return: shape(batch_size, choice_num, context_size+1, event_repr_size)
    """
    batch_size, num_choices, event_repr_size = choices.size()
    context_size = context.size(1)
    choices = choices.unsqueeze(2)
    context = context.unsqueeze(1).expand(
        batch_size, num_choices, context_size, event_repr_size)
    chain = torch.cat([context, choices], dim=2)
    return chain


class SCPredictorNetwork(nn.Module):
    """Chain model network."""

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 event_dim: int = 128,
                 seq_len: int = 8,
                 dropout: float = 0.1,
                 num_layers: int = 2,
                 num_heads: int = 16,
                 dim_feedforward: int = 1024,
                 directions: int = 1,
                 attention_func: str = "scaled-dot",
                 score_func: str = "euclidean",
                 pretrain_embedding=None):
        """Construction method for SCPredictor.

        Args:
            vocab_size (int): vocabulary size.
            embedding_dim (int): embedding dimension.
                Defaults to 300.
            event_dim (int): event dimension.
                Defaults to 128.
            seq_len (int): sequence length.
                Defaults to 8.
            dropout (float): dropout rate.
                Defaults to 0.1.
            num_layers (int): number of layers for transformer.
                Defaults to 2
            num_heads (int): number of heads for transformer.
                Defaults to 16.
            dim_feedforward (int): feedforward dimension for transformer.
                Defaults to 1024.
            directions (int): directions.
                Defaults to 1.
            attention_func (str): attention function name.
                Defaults to "scaled-dot".
            score_func (str): score function name.
                Defaults to "euclidean".
            pretrain_embedding: pretrained embedding.
                Defaults to None.
        """
        super(SCPredictorNetwork, self).__init__()
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout=dropout,
            pretrain_embedding=pretrain_embedding)
        self.event_encoder = EventFusionEncoder(
            embedding_dim=embedding_dim,
            event_dim=event_dim,
            dropout=dropout)
        self.sequence_model = Transformer(
            d_model=event_dim,
            seq_len=seq_len+1,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=dropout)
        self.attention = construct_attention_layer(
            func_name=attention_func,
            event_dim=event_dim,
            directions=directions)
        self.score = construct_score_layer(
            func_name=score_func,
            event_dim=event_dim,
            directions=directions)

    def forward(self, context: LongTensor, choices: LongTensor):
        """Forward.

        Args:
            context (LongTensor): context events.
                shape(batch_size, seq_len, 4)
            choices (LongTensor): choices events.
                shape(batch_size, num_choices, 4)
        """
        context_repr = self.event_encoder(self.embedding(context))
        choices_repr = self.event_encoder(self.embedding(choices))
        # chain_repr: shape(batch_size, choice_num, seq_len+1, event_repr_size)
        chain_repr = append_choices_to_context(context_repr, choices_repr)
        batch_size, num_choices, chain_len, _ = chain_repr.size()
        # seq_repr: shape(batch_size, choice_num, seq_len+1, event_repr_size)
        chain_repr = chain_repr.view(batch_size * num_choices, chain_len, -1)
        seq_repr = self.sequence_model(chain_repr)
        seq_repr = seq_repr.view(batch_size, num_choices, chain_len, -1)
        context = seq_repr[:, :, :-1, :]
        choices = seq_repr[:, :, -1:, :]
        # score: shape(batch_size, choice_num, seq_len)
        score = self.score(context, choices)
        # attention: shape(batch_size, choice_num, seq_len)
        attention = self.attention(context, choices)
        # score: shape(batch_size, choice_num)
        score = (score * attention).sum(-1)
        return score
