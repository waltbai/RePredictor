"""REPredictor network."""

from repredictor.network.attention import construct_attention_layer
from repredictor.network.event_encoder import construct_event_encoder
from repredictor.network.score import construct_score_layer
from repredictor.network.sequence_model import Transformer
import torch
from torch import FloatTensor, nn, LongTensor

from repredictor.network.embedding import Embedding


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


class REPredictorNetwork(nn.Module):
    """REPredictor network."""

    def __init__(self,
                 vocab_size: int,
                 role_size: int,
                 concept_size: int,
                 embedding_dim: int = 300,
                 event_dim: int = 128,
                 seq_len: int = 8,
                 dropout: float = 0.1,
                 hidden_dim_event: int = None,
                 num_layers_event: int = None,
                 num_heads_event: int = None,
                 dim_feedforward_event: int = None,
                 num_layers_seq: int = 2,
                 num_heads_seq: int = 16,
                 dim_feedforward_seq: int = 1024,
                 event_func: str = "rich-fusion",
                 attention_func: str = "scaled-dot",
                 score_func: str = "euclidean",
                 num_args: int = 23,
                 rich_event: bool = True,
                 use_concept: bool = True):
        """Construction method for REPredictor network.

        Args:
            vocab_size (int): vocabulary size.
            role_size (int): role type size.
            concept_size(int): entity type size.
            embedding_dim (int): embedding dimension.
                Defaults to 300.
            event_dim (int): event dimension.
                Defaults to 128.
            seq_len (int): sequence length.
                Defaults to 8.
            dropout (float): dropout rate.
                Defaults to 0.1.
            hidden_dim_event (int): hidden dimension for event fusion encoder.
                Defaults to None.
            num_layers_event (int): number of layers for event transformer.
                Defaults to None.
            num_heads_event (int): number of heads for event transformer.
                Defaults to None.
            dim_feedforward_event (int): feedforward dimension for event transformer.
                Defaults to None.
            num_layers_seq (int): number of layers for chain transformer.
                Defaults to 2.
            num_heads_seq (int): number of heads for chain transformer.
                Defaults to 16.
            dim_feedforward_seq (int): feedforward dimension for chain transformer.
                Defaults to 1024.
            event_func (str): event encoder to be used.
                Defaults to "rich-fusion"
            attention_func (str): attention function name.
                Defaults to "scaled-dot".
            score_func (str): score function name.
                Defaults to "euclidean".
            num_args (int): number of arguments.
                Defaults to 23.
            rich_event (bool): whether to use rich events.
                Defaults to True.
            use_concept (bool): whether to use entity types.
                Defaults to True.
        """
        super(REPredictorNetwork, self).__init__()
        self._num_args = num_args
        self._use_concept = use_concept
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            dropout=dropout)
        self.role_embedding = Embedding(
            vocab_size=role_size,
            embedding_dim=embedding_dim,
            dropout=dropout)
        self.concept_embedding = Embedding(
            vocab_size=concept_size,
            embedding_dim=embedding_dim,
            dropout=dropout)
        self.event_encoder = construct_event_encoder(
            func_name=event_func,
            embedding_dim=embedding_dim,
            event_dim=event_dim,
            dropout=dropout,
            hidden_dim=hidden_dim_event,
            num_layers=num_layers_event,
            num_heads=num_heads_event,
            dim_feedforward=dim_feedforward_event,
            num_args=num_args,
            use_concept=use_concept)
        self.sequence_model = Transformer(
            d_model=event_dim,
            seq_len=seq_len+1,
            nhead=num_heads_seq,
            num_layers=num_layers_seq,
            dim_feedforward=dim_feedforward_seq,
            dropout=dropout)
        self.attention = construct_attention_layer(
            func_name=attention_func,
            event_dim=event_dim)
        self.score = construct_score_layer(
            func_name=score_func,
            event_dim=event_dim)

    def encode_events(self, events: LongTensor) -> FloatTensor:
        """Encode events into dense representation

        Args:
            events (LongTensor): events, size(*, 2 + 2 * num_args).

        Returns:
            FloatTensor: encoded events, size(*, event_dim).
        """
        num_args = self._num_args
        # Split events
        verb, role, arg_role, arg_value, concept = \
            events.split([1, 1, num_args, num_args, num_args], dim=-1)
        # verb, role: size(*, )
        verb = verb.squeeze()
        role = role.squeeze()
        # prepare arg mask: size(*, num_args)
        arg_mask = (arg_role + arg_value) == 0
        # embedding
        verb_emb = self.embedding(verb)
        role_emb = self.role_embedding(role)
        arg_role_emb = self.role_embedding(arg_role)
        arg_value_emb = self.embedding(arg_value)
        concept_emb = self.concept_embedding(concept)
        # encode event
        return self.event_encoder(
            verb=verb_emb,
            role=role_emb,
            arg_role=arg_role_emb,
            arg_value=arg_value_emb,
            concept=concept_emb,
            arg_mask=arg_mask)

    def forward(self, context: LongTensor, choices: LongTensor) -> FloatTensor:
        """Forward.

        Args:
            context (LongTensor): context events.
                size(batch_size, context_size, 2 + 2 * num_args)
            choices (LongTensor): choices events.
                size(batch_size, num_choices, 2 + 2 * num_args)
        """
        # context_repr: shape(batch_size, context_size, event_repr_size)
        context_repr = self.encode_events(context)
        # choices_repr: shape(batch_size, num_choices, event_repr_size)
        choices_repr = self.encode_events(choices)
        # chain_repr: shape(batch_size, num_choices, context_size+1, event_repr_size)
        chain_repr = append_choices_to_context(context_repr, choices_repr)
        # hidden_repr: shape(batch_size, num_choices, context_size+1, event_repr_size)
        batch_size, num_choices, seq_len, _ = chain_repr.size()
        chain_repr = chain_repr.view(batch_size * num_choices, seq_len, -1)
        hidden_repr = self.sequence_model(chain_repr)
        hidden_repr = hidden_repr.view(batch_size, num_choices, seq_len, -1)
        # score: shape(batch_size, num_choices, context_size)
        context = hidden_repr[:, :, :-1, :]
        choices = hidden_repr[:, :, -1:, :]
        score = self.score(context, choices)
        # attention: shape(batch_size, num_choices, context_size)
        attention = self.attention(context, choices)
        # score: shape(batch_size, num_choices)
        score = (score * attention).sum(-1)
        return score
