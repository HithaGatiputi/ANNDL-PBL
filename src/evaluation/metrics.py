"""Classification and probability metrics, computed per subset of days."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, matthews_corrcoef

METRICS = ["acc", "f1", "mcc", "brier", "logloss", "ece"]


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def compute_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """All metrics for labels y and P(up) p (threshold 0.5)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    yhat = (p > 0.5).astype(int)
    if len(y) == 0:
        return {m: float("nan") for m in METRICS} | {"n": 0}
    return {"acc": accuracy_score(y, yhat), "f1": f1_score(y, yhat, zero_division=0),
            "mcc": matthews_corrcoef(y, yhat) if len(np.unique(yhat)) > 1 or len(np.unique(y)) > 1 else 0.0,
            "brier": brier_score_loss(y, p), "logloss": log_loss(y, p, labels=[0, 1]),
            "ece": ece(y, p, n_bins), "n": int(len(y))}


def subset_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Subsets of days: all, event days, post-event days, and each regime."""
    out = {"all": np.ones(len(df), bool),
           "event": df["is_event_day"].values.astype(bool),
           "post_event": df["is_post_event_day"].values.astype(bool)}
    for r in ["calm", "normal", "volatile", "crisis"]:
        out[r] = (df["regime"] == r).values
    return out


def metrics_by_subset(df: pd.DataFrame, n_bins: int = 10) -> dict[str, dict[str, float]]:
    """`df` has columns y, p_up, is_event_day, is_post_event_day, regime."""
    return {k: compute_metrics(df["y"].values[m], df["p_up"].values[m], n_bins)
            for k, m in subset_masks(df).items()}
