"""Embedding layers."""

from torch import nn, Tensor, LongTensor


__all__ = [
    "Embedding",
]


class Embedding(nn.Module):
    """Word embedding"""
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 300,
                 dropout: float = 0.,
                 pretrain_embedding: Tensor = None):
        """Construction method for embedding layer.

        Args:
            vocab_size (int): vocabulary size
            embedding_dim (int): embedding dimension
                Defaults to 300.
            dropout (float): dropout rate
                Defaults to 0.
            pretrain_embedding (Tensor):
                Defaults to None.
        """
        super(Embedding, self).__init__()
        # Define word embedding
        if pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrain_embedding,
                padding_idx=0)
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0)
        # Define dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, indices: LongTensor):
        """Forward.

        Args:
            indices (LongTensor): indices
        """
        output = self.dropout(self.embedding(indices))
        return output
