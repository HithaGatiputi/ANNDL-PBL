"""Fixed, time-ordered train / val / test splits (never shuffled across splits)."""
from __future__ import annotations

import pandas as pd


def split_of(dates: pd.DatetimeIndex | pd.Series, cfg: dict) -> pd.Series:
    """Label each date 'train', 'val' or 'test' (or 'out' if after the test end)."""
    d = pd.to_datetime(pd.Series(dates))
    tr, va, te = (pd.Timestamp(cfg["split"][k]) for k in ["train_end", "val_end", "test_end"])
    lab = pd.Series("out", index=d.index)
    lab[d <= te] = "test"
    lab[d <= va] = "val"
    lab[d <= tr] = "train"
    lab[d < pd.Timestamp(cfg["data"]["start"])] = "warmup"
    return lab.values


def split_mask(dates, cfg: dict, name: str):
    """Boolean mask of dates belonging to split `name`."""
    return split_of(dates, cfg) == name
