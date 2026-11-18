"""Dynamic graph layer for the 5-asset graph, written directly in PyTorch (no torch-geometric).

Edge logits = A_static (learned, shared) + gate * Delta A (from the event) + GAT attention scores.
A = row-softmax of the logits (self-loops included), then two residual graph layers
h <- ELU(A @ W h) + h.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATScores(nn.Module):
    """GAT-style scores e_ij = LeakyReLU(a^T [W h_i || W h_j])."""

    def __init__(self, dim: int, slope: float):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Linear(dim, 1, bias=False)  # a = [a_src ; a_dst]
        self.a_dst = nn.Linear(dim, 1, bias=False)
        self.slope = slope

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, d) -> scores (B, N, N)."""
        wh = self.W(h)
        return F.leaky_relu(self.a_src(wh) + self.a_dst(wh).transpose(1, 2), self.slope)


class DynamicGraph(nn.Module):
    """Builds A_tau and applies the graph layers to node states."""

    def __init__(self, n_nodes: int, dim: int, n_layers: int, dropout: float, slope: float):
        super().__init__()
        self.A_static = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.att = GATScores(dim, slope)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)

    def adjacency(self, h: torch.Tensor, delta: torch.Tensor | None, gate: torch.Tensor | None) -> torch.Tensor:
        """Row-normalised adjacency (B, N, N); every row sums to 1."""
        logits = self.A_static + self.att(h)
        if delta is not None:
            logits = logits + gate * delta
        return torch.softmax(logits, dim=-1)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Two residual message-passing layers: h <- ELU(A @ W h) + h."""
        for lin in self.layers:
            h = self.drop(F.elu(A @ lin(h))) + h
        return h
