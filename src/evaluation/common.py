"""Helpers shared by the evaluation scripts: load trained models and seed-ensembled predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.training.dataset import DayData, build_dataset
from src.training.train import build_model, to_batch
from src.utils import paths
from src.utils.config import load_config

MAIN_MODEL = "rcdgnn_full"


def available(variants: list[str]) -> list[str]:
    """Variants that have saved results."""
    return [v for v in variants if (paths.RESULTS / v).exists()]


def load_models(variant: str) -> tuple[dict, DayData, list[torch.nn.Module]]:
    """Config, dataset and the trained model of every seed (eval mode)."""
    cfg = load_config(variant)
    d = build_dataset(cfg)
    models = []
    for f in sorted((paths.RESULTS / variant).glob("seed*/model.pt")):
        m = build_model(cfg, d)
        m.load_state_dict(torch.load(f))
        models.append(m.eval())
    return cfg, d, models


@torch.no_grad()
def forward_days(models, d: DayData, idx: np.ndarray, **flags) -> dict[str, np.ndarray]:
    """Deterministic forward pass, averaged over seeds: p_up, A_last, gate_last."""
    b = to_batch(d, idx) | flags
    outs = [m(b) for m in models]
    return {"p_up": np.mean([torch.sigmoid(o["logit"]).numpy() for o in outs], 0),
            "A": np.mean([o["A_last"].numpy() for o in outs], 0),
            "gate": np.mean([o["gate_last"].numpy() for o in outs], 0)}


def seed_preds(variant: str, split: str = "test") -> list[pd.DataFrame]:
    """Prediction frames of every seed."""
    return [pd.read_csv(f, parse_dates=["date"]) for f in sorted((paths.RESULTS / variant).glob(f"seed*/preds_{split}.csv"))]
