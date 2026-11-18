"""Edge generator: turns the (decayed) reaction signature + event embedding of a day into a
change of the 5x5 edge logits (Delta A) and a gate saying how much the event changes the graph."""
from __future__ import annotations

import torch
import torch.nn as nn


class EdgeGenerator(nn.Module):
    """MLP [S_tau, E_tau] -> 32 -> N*N (tanh) and gate = sigmoid(w . [S_tau, E_tau] + b)."""

    def __init__(self, in_dim: int, hidden: int, n_nodes: int):
        super().__init__()
        self.n = n_nodes
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_nodes * n_nodes), nn.Tanh())
        self.gate = nn.Linear(in_dim, 1)

    def forward(self, se: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """se: (..., D) -> delta_a (..., N, N) in [-1, 1], gate (..., 1, 1) in [0, 1]."""
        delta = self.mlp(se).reshape(*se.shape[:-1], self.n, self.n)
        gate = torch.sigmoid(self.gate(se)).unsqueeze(-1)
        return delta, gate
