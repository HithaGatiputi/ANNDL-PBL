"""Prediction head: MLP -> logit. Dropout here is kept on at inference for MC-dropout uncertainty."""
import torch
import torch.nn as nn


class DirectionHead(nn.Module):
    """MLP(in_dim -> hidden -> 1) with dropout."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> logit (B,)."""
        return self.net(x).squeeze(-1)
