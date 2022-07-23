"""Event encoder layers."""
import torch
from torch import nn, FloatTensor, BoolTensor


__all__ = [
    "EventFusionEncoder",
    "RichEventFusionEncoder",
    "RichEventTransformerEncoder",
    "construct_event_encoder",
]


# For SCPredictor
class EventFusionEncoder(nn.Module):
    """Event encoder."""

    def __init__(self,
                 embedding_dim: int = 300,
                 event_dim: int = 128,
                 num_components: int = 4,
                 dropout: float = 0.):
        """Construction method for event fusion encoder.

        Args:
            embedding_dim (int): embedding dimension.
                Defaults to 300.
            event_dim (int): event dimension.
                Defaults to 128.
            num_components (int): number of components in event.
                Defaults to 4.
            dropout (float): dropout rate.
                Defaults to 0.
        """
        super(EventFusionEncoder, self).__init__()
        self._num_components = num_components
        self.linear = nn.Linear(
            in_features=embedding_dim * num_components,
            out_features=event_dim,
            bias=True)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, events: FloatTensor):
        """Forward.

        Args:
            events (FloatTensor): shape(*, num_components, embedding_size)

        Returns:
            FloatTensor: (*, event_repr_size)
        """
        shape = events.size()
        input_shape = shape[:-2] + (shape[-1] * self._num_components, )
        projections = self.activation(self.linear(events.view(input_shape)))
        projections = self.dropout(projections)
        return projections


# For rich event representation
class RichEventFusionEncoder(nn.Module):
    """Rich event fusion encoder."""

    def __init__(self,
                 embedding_dim: int = 300,
                 event_dim: int = 128,
                 hidden_dim: int = 300,
                 dropout: float = 0.,
                 use_concept: bool = False):
        """Construction method for rich event fusion encoder.

        Args:
            embedding_dim (int): embedding dimension.
                Defaults to 300.
            event_dim (int): event dimension.
                Defaults to 128.
            hidden_dim (int): hidden dimension for arguments.
                Defaults to 300.
            dropout (float): dropout rate.
                Defaults to 0.
            use_concept (bool): if entity type is used.
                Defaults to False.
        """
        super(RichEventFusionEncoder, self).__init__()
        input_dim = embedding_dim * 3 if use_concept else embedding_dim * 2
        self._use_concept = use_concept
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(embedding_dim * 2 + hidden_dim, event_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, verb: FloatTensor, role: FloatTensor,
                arg_role: FloatTensor, arg_value: FloatTensor, concept: FloatTensor,
                arg_mask: BoolTensor = None):
        """Compute event representation.

        Args:
            verb (FloatTensor): verb, shape(*, embedding_dim).
            role (FloatTensor): protagonist role, shape(*, embedding_dim).
            arg_role (FloatTensor): arg role, shape(*, num_args, embedding_dim).
            arg_value (FloatTensor): arg value, shape(*, num_args, embedding_dim).
            concept (FloatTensor): arg type, shape(*, num_args, embedding_dim).
            arg_mask (BoolTensor): arg mask, shape(*, num_args).
                Defaults to None.

        Returns:
            FloatTensor: event representation, shape(*, event_dim)
        """
        # prepare real_mask: shape(*, num_args, 1)
        real_mask = arg_role.new_ones(arg_mask.size(), dtype=torch.float)
        real_mask = real_mask.masked_fill(arg_mask, 0.).unsqueeze(len(arg_mask.size()))
        # calculate arg_repr: shape(*, num_args, hidden_dim)
        if self._use_concept:
            args = torch.cat([arg_role, arg_value, concept], dim=-1)
        else:
            args = torch.cat([arg_role, arg_value], dim=-1)
        args_repr = self.linear1(args) * real_mask
        # args: shape(*, hidden_dim)
        # sum along the arg_num dimension
        args = args_repr.sum(dim=-2)
        # event: shape(*, embedding_dim * 2 + hidden_dim)
        event = torch.cat([verb, role, args], dim=-1)
        projections = self.activation(self.linear2(event))
        projections = self.dropout(projections)
        return projections


class RichEventTransformerEncoder(nn.Module):
    """Rich event transformer encoder."""

    def __init__(self,
                 embedding_dim: int = 300,
                 event_dim: int = 128,
                 num_args: int = 25,
                 num_layers: int = 1,
                 num_heads: int = 8,
                 dim_feedforward: int = 512,
                 dropout: float = 0.,
                 use_concept: bool = False):
        """Construction method for rich event transformer encoder.

        Args:
            embedding_dim (int): embedding dimension.
                Defaults to 300.
            event_dim (int): event dimension.
                Defaults to 128.
            num_args (int): max number of arguments.
                Defaults to 23.
            num_layers (int): number of transformer layers.
                Defaults to 1.
            num_heads (int): number of transformer heads.
                Defaults to 8.
            dim_feedforward (int): transformer feedforward dimension.
                Defaults to 512.
            dropout (float): dropout rate.
                Defaults to 0.
            use_concept (bool): if entity type is used.
                Defaults to False.
        """
        super(RichEventTransformerEncoder, self).__init__()
        self._num_args = num_args
        self._event_dim = event_dim
        self._use_concept = use_concept
        # map input
        self.pred_proj = nn.Linear(embedding_dim * 2, event_dim)
        input_dim = embedding_dim * 3 if use_concept else embedding_dim * 2
        self.arg_proj = nn.Linear(input_dim, event_dim)
        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=event_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers)

    def forward(self, verb: FloatTensor, role: FloatTensor,
                arg_role: FloatTensor, arg_value: FloatTensor, concept: FloatTensor,
                arg_mask: BoolTensor = None):
        """Compute event representation.

        Args:
            verb (FloatTensor): verb, shape(*, embedding_dim).
            role (FloatTensor): protagonist role, shape(*, embedding_dim).
            arg_role (FloatTensor): arg role, shape(*, num_args, embedding_dim).
            arg_value (FloatTensor): arg value, shape(*, num_args, embedding_dim).
            concept (FloatTensor): arg type, shape(*, num_args, embedding_dim).
            arg_mask (BoolTensor): arg mask, shape(*, num_args).
                Defaults to None.

        Returns:
            FloatTensor: event representation, shape(*, event_dim)
        """
        # seq: shape(*, num_args+1, event_dim)
        dims = len(verb.size())
        p_repr = self.pred_proj(torch.cat([verb, role], dim=-1).unsqueeze(dims-1))
        if self._use_concept:
            arg_repr = self.arg_proj(torch.cat([arg_role, arg_value, concept], dim=-1))
        else:
            arg_repr = self.arg_proj(torch.cat([arg_role, arg_value], dim=-1))
        seq = torch.cat([p_repr, arg_repr], dim=-2)
        # mask: shape(*, num_args+1)
        size = verb.size()
        mask = torch.cat([
            arg_mask.new_zeros(size[:-1] + (1, ), dtype=torch.bool),
            arg_mask,
        ], dim=-1)
        # hidden: shape(*, num_args+1, evnet_dim)
        hidden = self.transformer(
            seq.view(-1, seq.size(-2), seq.size(-1)),
            src_key_padding_mask=mask.view(-1, mask.size(-1)))
        hidden = hidden.view(seq.size())
        # event_repr: shape(*, event_dim)
        event_repr = hidden.select(dim=-2, index=0)
        return event_repr


def construct_event_encoder(func_name: str,
                            embedding_dim: int = 300,
                            event_dim: int = 128,
                            dropout: float = 0.1,
                            hidden_dim: int = 300,
                            num_layers: int = 1,
                            num_heads: int = 8,
                            dim_feedforward: int = 512,
                            num_args: int = 23,
                            use_concept: bool = False):
    """Construct event encoder.

    Args:
        func_name (str): event encoder name.
            "fusion", "rich-fusion", or "rich-trans".
        embedding_dim (int, optional): word embedding dimension.
            Defaults to 300.
        event_dim (int, optional): event embedding dimension.
            Defaults to 128.
        dropout (float, optional): dropout rate.
            Defaults to 0.1.
        hidden_dim (int, optional): in case needed.
            Defaults to 300.
        num_layers (int, optional): in case needed.
            Defaults to 1.
        num_heads (int, optional): in case needed.
            Defaults to 8.
        dim_feedforward (int, optional): in case needed.
            Defaults to 512.
        num_args (int, optional): number of arguments.
            Defaults to 23.
        use_concept (bool, optional): whether to use entity type.
            Defaults to False.
    """
    if func_name == "fusion":
        return EventFusionEncoder(
            embedding_dim=embedding_dim,
            event_dim=event_dim,
            dropout=dropout)
    elif func_name == "rich-fusion":
        return RichEventFusionEncoder(
            embedding_dim=embedding_dim,
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_concept=use_concept)
    elif func_name == "rich-trans":
        return RichEventTransformerEncoder(
            embedding_dim=embedding_dim,
            event_dim=event_dim,
            num_args=num_args,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_concept=use_concept)
    else:
        raise KeyError(f"Unknown event encoder '{func_name}'!")
