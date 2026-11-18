"""Build model-ready arrays for every trading day t in the modelling period.

For day t:
  X_nodes : (L, N=5, F=3)  per-node [return or yield change, 5-day rolling return, 20-day vol]
                           for days t-L+1 .. t (inputs may reach back into the previous split)
  S       : (L, D_s)       decayed reaction signature for each day in the window
  E       : (L, D_e)       decayed event embedding + sentiment for each day in the window
  G       : (G_dim,)       global features of day t: RSI, MACD hist, VIX, VIX change, regime one-hot
  y       : 1 if gold closes higher on t+1
Every feature row for day t uses only data dated <= t (all daily inputs are timestamped by 17:00 ET).
Scalers are fitted on train-split rows only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.technical import macd_hist, rsi
from src.training.splits import split_of
from src.utils import paths

NODE_COLS = ["gold_ret", "usd_ret", "bonds_dchg", "eq_ret", "oil_ret"]
REGIMES = ["calm", "normal", "volatile", "crisis"]


@dataclass
class DayData:
    """All arrays for the modelling period, aligned on `meta.index`."""
    X: np.ndarray        # (T, L, N, F)
    S: np.ndarray        # (T, L, D_s)
    E: np.ndarray        # (T, L, D_e)
    G: np.ndarray        # (T, G_dim)
    y: np.ndarray        # (T,)
    meta: pd.DataFrame   # date, y, split, is_event_day, is_post_event_day, regime
    s_cols: list
    e_cols: list
    g_cols: list


def node_feature_table(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-day node features (columns '<node>_ret', '<node>_r5', '<node>_vol'), causal rolling only."""
    f = cfg["features"]
    out = {}
    for col in NODE_COLS:
        node = col.split("_")[0]
        out[f"{node}_x"] = daily[col]
        out[f"{node}_r5"] = daily[col].rolling(f["roll_short"]).sum()
        out[f"{node}_vol"] = daily[col].rolling(f["roll_vol"]).std()
    return pd.DataFrame(out, index=daily.index)


def global_feature_table(daily: pd.DataFrame, regimes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Global features for day t (technicals on gold, VIX, regime one-hot)."""
    f = cfg["features"]
    g = pd.DataFrame(index=daily.index)
    g["rsi"] = rsi(daily["gold_close"], f["rsi_period"])
    g["macd_hist"] = macd_hist(daily["gold_close"], f["macd_fast"], f["macd_slow"], f["macd_signal"])
    g["vix"] = daily["vix"]
    g["vix_chg"] = daily["vix_chg"]
    reg = regimes["regime_vol"].reindex(daily.index)
    for r in REGIMES:
        g[f"reg_{r}"] = (reg == r).astype(float)
    return g


def event_flags(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """is_event_day (an event is attached to t) and is_post_event_day (an event was on t-1)."""
    ev = pd.read_csv(paths.EVENTS_CSV)
    ev_days = set(pd.to_datetime(ev["trading_day"]))
    is_ev = np.array([d in ev_days for d in index])
    is_post = np.r_[False, is_ev[:-1]]
    return is_ev, is_post


def _load_daily_aligned(file: str, index: pd.DatetimeIndex) -> tuple[np.ndarray, list]:
    """Load a per-trading-day feature parquet (signature/embedding), zero-filled on missing days."""
    p = paths.PROCESSED / file
    if not p.exists():
        return np.zeros((len(index), 0), dtype=np.float32), []
    df = pd.read_parquet(p).reindex(index).fillna(0.0)
    df = df.select_dtypes(include=[np.number])
    return df.values.astype(np.float32), list(df.columns)


def build_dataset(cfg: dict, daily_ext: pd.DataFrame | None = None, scale: bool = True) -> DayData:
    """Assemble windows for all modelling days. `daily_ext` can be passed (used by leakage tests)."""
    if daily_ext is None:
        daily_ext = pd.read_parquet(paths.INTERIM / "daily_ext.parquet")
    regimes = pd.read_parquet(paths.REGIMES_PARQUET)
    L = cfg["model"]["lookback"]
    nodes = node_feature_table(daily_ext, cfg)
    glob = global_feature_table(daily_ext, regimes, cfg)
    split = split_of(daily_ext.index, cfg)

    if scale:  # fit on train rows only, apply everywhere
        tr = (split == "train")
        node_sc = StandardScaler().fit(nodes[tr].dropna())
        nodes = pd.DataFrame(node_sc.transform(nodes), index=nodes.index, columns=nodes.columns)
        num_g = ["rsi", "macd_hist", "vix", "vix_chg"]
        g_sc = StandardScaler().fit(glob.loc[tr, num_g].dropna())
        glob[num_g] = g_sc.transform(glob[num_g])

    sig_file = cfg["model"].get("signature_file", "signatures_daily.parquet")
    S_all, s_cols = _load_daily_aligned(sig_file, daily_ext.index)
    E_all, e_cols = _load_daily_aligned("event_emb_daily.parquet", daily_ext.index)

    node_arr = nodes.values.reshape(len(nodes), len(NODE_COLS), 3).astype(np.float32)
    start, end = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    pos = np.where((daily_ext.index >= start) & (daily_ext.index <= end))[0]
    pos = pos[pos >= L - 1]
    X = np.stack([node_arr[p - L + 1:p + 1] for p in pos])
    S = np.stack([S_all[p - L + 1:p + 1] for p in pos])
    E = np.stack([E_all[p - L + 1:p + 1] for p in pos])
    G = glob.values[pos].astype(np.float32)
    y = daily_ext["y_next"].values[pos].astype(np.float32)
    idx = daily_ext.index[pos]
    is_ev, is_post = event_flags(daily_ext.index)
    meta = pd.DataFrame({"date": idx, "y": y.astype(int), "split": split[pos],
                         "is_event_day": is_ev[pos], "is_post_event_day": is_post[pos],
                         "regime": regimes["regime_vol"].reindex(idx).values})
    if np.isnan(X).any() or np.isnan(G).any():
        raise ValueError("NaN in features: not enough warm-up history before the first modelling day")
    return DayData(X, S, E, G, y, meta, s_cols, e_cols, list(glob.columns))
