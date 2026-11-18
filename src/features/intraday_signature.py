"""Intraday (1-minute) reaction signatures, falling back to the daily proxy when data are poor.

Window per event: 60 minutes before the release to 17:00 ET on its trading day.
Gold window flagged (-> daily signature, has_intraday = 0) if > max_missing_frac of the minutes
are missing or there are fewer than 30 pre-event bars (e.g. a release while the market is shut).
A cross asset with a poor window keeps its daily-proxy value. Bonds stay daily (no free intraday yields).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.download_intraday import event_window_utc
from src.features.reaction_signature import CROSS, compute_signature, event_signatures_daily
from src.utils import paths

MIN_PRE_BARS = 30


def load_window(event_id: str, asset: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Close prices indexed by 1-min bar END time (UTC), restricted to [start, end]."""
    f = paths.RAW_INTRADAY / f"{event_id}_{asset}.csv"
    if not f.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(f)
    if df.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True) + pd.Timedelta(minutes=1)
    s = pd.Series(df["close"].values.astype(float), index=idx)
    return s[(s.index >= start) & (s.index <= end)]


def window_quality(s: pd.Series, start, end, event_time) -> tuple[float, int]:
    """(fraction of minutes missing, number of pre-event bars)."""
    expected = int((end - start).total_seconds() // 60)
    return 1 - len(s) / max(expected, 1), int((s.index <= event_time).sum())


def event_signatures_intraday(events: pd.DataFrame, cfg: dict, return_quality: bool = False):
    """Raw signature per event (same columns as the daily version)."""
    daily = event_signatures_daily(events, cfg).set_index("event_id")
    max_miss = cfg["reaction"]["max_missing_frac"]
    pre_min = cfg["reaction"]["pre_event_min"]
    rows, qual = [], []
    for _, ev in events.iterrows():
        start, end = event_window_utc(ev, pre_min)
        et = pd.Timestamp(ev["datetime_utc"])
        win = {a: load_window(ev["event_id"], a, start, end) for a in ["gold"] + CROSS}
        q = {a: window_quality(s, start, end, et) for a, s in win.items()}
        ok = {a: (q[a][0] <= max_miss and q[a][1] >= MIN_PRE_BARS) for a in win}
        qual.append({"event_id": ev["event_id"], "type": ev["type"], "year": ev["trading_day"][:4],
                     **{f"miss_{a}": q[a][0] for a in win}, **{f"ok_{a}": ok[a] for a in win}})
        base = daily.loc[ev["event_id"]].to_dict() if ev["event_id"] in daily.index else None
        if not ok["gold"]:
            if base is not None:
                rows.append({"event_id": ev["event_id"], **base})
            continue
        pre_vol = {}
        for a, s in win.items():
            r = np.log(s[s.index <= et]).diff().dropna()
            pre_vol[a] = float(r.std()) if ok[a] and len(r) > 5 else 0.0
        sig = compute_signature({a: s for a, s in win.items() if ok[a]}, et, pre_vol, ["5m", "1h", "close"], None)
        if base is not None:  # cross assets with poor intraday data keep the daily proxy
            for a in CROSS:
                if not ok[a]:
                    sig[f"xa_{a}_dir"], sig[f"xa_{a}_mag"] = base[f"xa_{a}_dir"], base[f"xa_{a}_mag"]
        rows.append({"event_id": ev["event_id"], **sig})
    out = pd.DataFrame(rows)
    pd.DataFrame(qual).to_csv(paths.PROCESSED / "intraday_quality.csv", index=False)
    return (out, pd.DataFrame(qual)) if return_quality else out
