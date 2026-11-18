"""Temporal module: LSTM over the per-day graph read-outs z_1..z_L."""
import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """LSTM(in_dim -> hidden); returns the last hidden state."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, L, in_dim) -> (B, hidden)."""
        _, (h, _) = self.lstm(z)
        return h[-1]
