"""RC-DGNN: Reaction-Conditioned Dynamic Graph Neural Network.

For each day tau of the 30-day window:
  h_tau     = Linear(F -> 32)(X_nodes[:, tau])                      node encoding (B, 5, 32)
  dA, gate  = EdgeGenerator([S_tau, E_tau])                         event-driven edge change
  A_tau     = softmax_rows(A_static + gate * dA + GAT(h_tau))
  h_tau     = 2 x [ELU(A_tau @ W h_tau) + h_tau]                    residual graph layers
  z_tau     = [h_tau[gold], mean_nodes(h_tau)]                      (B, 64)
LSTM(64 -> 64) over z_1..z_L, head MLP([lstm_out, G, E_t] -> 32 -> 1).
Static variant (reaction_conditioned = false): dA = 0 and gate = 0.
Penalty: L1 on the effective edge change gate * dA.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.dynamic_gnn import DynamicGraph
from src.models.edge_generator import EdgeGenerator
from src.models.heads import DirectionHead
from src.models.temporal import TemporalLSTM

GOLD = 0


class RCDGNN(nn.Module):
    """Full model; see module docstring."""

    def __init__(self, cfg: dict, f_dim: int, s_dim: int, e_dim: int, g_dim: int, n_nodes: int = 5):
        super().__init__()
        m = cfg["model"]
        d = m["node_hidden"]
        self.rc = bool(m["reaction_conditioned"])
        self.use_e = bool(m["use_event_emb"])
        self.l1 = float(m["l1_delta_a"])
        self.node_enc = nn.Linear(f_dim, d)
        self.graph = DynamicGraph(n_nodes, d, m["gnn_layers"], m["gnn_dropout"], m["leaky_slope"])
        self.edge_gen = EdgeGenerator(s_dim + e_dim, m["edge_hidden"], n_nodes) if self.rc else None
        self.temporal = TemporalLSTM(2 * d, m["hidden"])
        self.drop = nn.Dropout(m["dropout"])
        self.e_dim = e_dim if self.use_e else 0
        self.head = DirectionHead(m["hidden"] + g_dim + self.e_dim, m["head_hidden"], m["dropout"])

    def forward(self, batch: dict) -> dict:
        """batch: X (B,L,N,F), S (B,L,Ds), E (B,L,De), G (B,Gd) -> logit, L1 penalty, last-day A, gate."""
        X, S, E, G = batch["X"], batch["S"], batch["E"], batch["G"]
        B, L, N, F = X.shape
        h = self.node_enc(X.reshape(B * L, N, F))                    # all days at once
        delta = gate = None
        # batch["no_event_edges"] = True gives the counterfactual graph without the event change
        if self.rc and not batch.get("no_event_edges", False):
            se = torch.cat([S, E], dim=-1).reshape(B * L, -1)
            delta, gate = self.edge_gen(se)
        A = self.graph.adjacency(h, delta, gate)                      # (B*L, N, N)
        h = self.graph(h, A)
        z = torch.cat([h[:, GOLD], h.mean(1)], dim=-1).reshape(B, L, -1)
        out = self.drop(self.temporal(z))
        parts = [out, G] + ([E[:, -1]] if self.use_e else [])
        logit = self.head(torch.cat(parts, dim=-1))
        A = A.reshape(B, L, N, N)
        if delta is not None:
            eff = (gate * delta).reshape(B, L, N, N)
            penalty = self.l1 * eff.abs().mean()
            gate_last = gate.reshape(B, L)[:, -1]
        else:
            penalty = torch.zeros((), device=X.device)
            gate_last = torch.zeros(B, device=X.device)
        return {"logit": logit, "penalty": penalty, "A_last": A[:, -1], "gate_last": gate_last}
