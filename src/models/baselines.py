"""Baseline models: majority class, logistic regression, and LSTM (optionally + event embedding)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

GOLD = 0  # node index of gold in X_nodes


class MajorityBaseline:
    """Always predicts the train-split majority class; P(up) = train up-rate."""

    def fit(self, y_train: np.ndarray) -> "MajorityBaseline":
        """Store the train up-rate."""
        self.rate = float(np.mean(y_train))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        """Constant P(up) for n days."""
        return np.full(n, self.rate)


def logreg_features(X: np.ndarray, G: np.ndarray, lags: int) -> np.ndarray:
    """Gold's last `lags` (scaled) returns from the window + the global features of day t."""
    return np.concatenate([X[:, -lags:, GOLD, 0], G], axis=1)


class LogRegBaseline:
    """Logistic regression on gold's last 5 returns + global features."""

    def __init__(self, C: float, lags: int):
        self.lags = lags
        self.model = LogisticRegression(C=C, max_iter=2000)

    def fit(self, X, G, y) -> "LogRegBaseline":
        """Fit on train windows X, global features G and labels y."""
        self.model.fit(logreg_features(X, G, self.lags), y)
        return self

    def predict_proba(self, X, G) -> np.ndarray:
        """P(up) for each sample."""
        return self.model.predict_proba(logreg_features(X, G, self.lags))[:, 1]


class LSTMBaseline(nn.Module):
    """LSTM over the gold node's features; the news variant adds the day-t event embedding to the head."""

    def __init__(self, n_feat: int, hidden: int, dropout: float, e_dim: int = 0):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.e_dim = e_dim
        self.head = nn.Linear(hidden + e_dim, 1)

    def forward(self, batch: dict) -> dict:
        """batch with X (B,L,N,F) and E -> logit (B,) and a zero penalty."""
        x = batch["X"][:, :, GOLD, :]                # (B, L, F)
        _, (h, _) = self.lstm(x)
        z = h[-1]
        if self.e_dim:
            z = torch.cat([z, batch["E"][:, -1, :]], dim=1)  # event embedding of day t (decayed)
        return {"logit": self.head(self.drop(z)).squeeze(-1), "penalty": torch.zeros(())}
