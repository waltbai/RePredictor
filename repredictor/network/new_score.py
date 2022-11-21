from torch import nn

from repredictor.network.sequence_model import PositionEncoder


class TransformerScore(nn.Module):
    """Transformer encoder-decoder structure for scoring."""

    def __init__(self,
                 event_dim=128,
                 num_heads=16,
                 dim_feedforward=1024,
                 num_encoder_layers=2,
                 num_decoder_layers=1,
                 dropout=0.1,
                 seq_len=8):
        super(TransformerScore, self).__init__()
        self.transformer = nn.Transformer(
            d_model=event_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True)
        self.pos_enc = PositionEncoder(
            d_model=event_dim,
            seq_len=seq_len,
            dropout=dropout)
        self.linear = nn.Linear(event_dim, 1)

    def forward(self, context, choices):
        """Forward.

        Args:
            context (FloatTensor): context events,
                size(batch_size, seq_len, event_dim)
            choices (FloatTensor): choices events,
                size(batch_size, num_choices, event_dim)

        Returns:
            FloatTensor: score, size(batch_size, num_choices)
        """
        context = self.pos_enc(context)
        hidden = self.transformer(context, choices)
        score = self.linear(hidden)
        score = score.view(score.size()[:-1])
        return score
