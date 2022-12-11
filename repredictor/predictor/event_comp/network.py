"""EventComp network."""
import torch
from torch import nn, LongTensor, FloatTensor

from repredictor.network.embedding import Embedding


class EventCompNetwork(nn.Module):
    """EventComp network."""

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 arg_comp_hidden_dim: int = 600,
                 event_dim: int = 300,
                 event_comp_hidden_1_dim: int = 400,
                 event_comp_hidden_2_dim: int = 200,
                 dropout: float = 0.3):
        """Construction method for EventComp.

        Args:
            vocab_size (int): vocabulary size.
            embedding_dim (int): word embedding dimension.
                Defaults to 300.
            arg_comp_hidden_dim (int): arg comp hidden layer dimension.
                Defaults to 600.
            event_dim (int): event embedding dimension.
                Defaults to 300.
            event_comp_hidden_1_dim (int): event comp hidden layer 1 dimension.
                Defaults to 400.
            event_comp_hidden_2_dim (int): event comp hidden layer 2 dimension.
                Defaults to 200.
            dropout (float): dropout rate.
                Defaults to 0.3.
        """
        super(EventCompNetwork, self).__init__()
        # Word embedding
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim)
        # Argument composition
        self.arg_comp_1 = nn.Linear(embedding_dim * 4, arg_comp_hidden_dim)
        self.arg_comp_2 = nn.Linear(arg_comp_hidden_dim, event_dim)
        # Event coherence
        self.event_comp_1 = nn.Linear(
            event_dim * 2, event_comp_hidden_1_dim)
        self.event_comp_2 = nn.Linear(
            event_comp_hidden_1_dim, event_comp_hidden_2_dim)
        self.event_comp_3 = nn.Linear(
            event_comp_hidden_2_dim, 1)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def arg_comp(self, event: LongTensor) -> FloatTensor:
        """Arg composition for a single event.

        Args:
            event (LongTensor): event, size(batch_size, *, 4)

        Returns:
            FloatTensor: event representation,
                size(batch_size, *, arg_comp_hidden_2_dim)
        """
        event_emb = self.dropout(self.embedding(event))
        shape = event_emb.size()
        input_shape = shape[:-2] + (shape[-1] * 4,)
        event_emb = event_emb.view(input_shape)
        hidden_1 = self.dropout(self.activation(self.arg_comp_1(event_emb)))
        hidden_2 = self.arg_comp_2(hidden_1)
        return hidden_2

    def forward_pair(self, e1: LongTensor, e2: LongTensor) -> FloatTensor:
        """Compute coherence of a event pair.

        Args:
            e1 (LongTensor): event 1, size(batch_size, *, 4)
            e2 (LongTensor): event 2, size(batch_size, *, 4)

        Returns:
            FloatTensor: score, size(batch_size, *)
        """
        e1_repr = self.arg_comp(e1)
        e2_repr = self.arg_comp(e2)
        pair_repr = torch.cat([e1_repr, e2_repr], dim=-1)
        hidden_1 = self.activation(self.event_comp_1(pair_repr))
        hidden_2 = self.activation(self.event_comp_2(hidden_1))
        output = self.event_comp_3(hidden_2).squeeze()
        return self.sigmoid(output)

    def forward_chain(self, context: LongTensor, choices: LongTensor) -> FloatTensor:
        """Compute the coherence of each choice.

        Args:
            context (LongTensor): context events.
                size(batch_size, context_size, 4)
            choices (LongTensor): choices events.
                size(batch_size, num_choices, 4)

        Returns:
            FloatTensor: Coherence of each choice.
        """
        batch_size = context.size(0)
        context_size = context.size(1)
        num_choices = choices.size(1)
        # Convert context to size(batch_size, num_choices, context_size, 4)
        context = context.unsqueeze(1)
        context = context.expand(batch_size, num_choices, context_size, 4)
        # Convert choices to size(batch_size, num_choices, context_size, 4)
        choices = choices.unsqueeze(2)
        choices = choices.expand(batch_size, num_choices, context_size, 4)
        # Compute pairwise coherence, size(batch_size, num_choices, context_size)
        pair_score = self.forward_pair(context, choices)
        # Average along the context_size dimension
        score = pair_score.mean(dim=-1)
        return score
