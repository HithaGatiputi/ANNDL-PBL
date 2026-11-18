"""Market Reaction Signature: how gold and other assets actually reacted to each event.

One function, `compute_signature`, works on both intraday (1-minute) and daily price paths.
Price paths are pandas Series of prices indexed by the time the price was observed (UTC);
for 1-minute bars that is the bar END time, so p(t) = last price with index <= t.

Magnitudes are moves divided by the pre-event volatility scaled to the horizon
(sigma * sqrt(steps)), so intraday (sigma per minute) and daily (sigma per day) magnitudes are in
the same unit. Expected reactions and imputation values come from TRAIN-split events only.

CLI:  python -m src.features.reaction_signature --mode daily|intraday
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.training.splits import split_of
from src.utils import paths
from src.utils.config import load_config

ASSETS = ["gold", "usd", "equities", "oil"]
CROSS = ["usd", "equities", "oil"]
TYPES = ["FOMC", "CPI", "NFP"]
CHECKPOINT_NAMES = {5: "5m", 60: "1h"}
# continuous features that are scaled by their train-event std (no centering, so 0 = no reaction)
CONT = ["latency_min", "mag_5m", "mag_1h", "mag_close", "vol_ratio", "persistence", "gap_gold"] + \
       [f"xa_{a}_mag" for a in CROSS]
MAYBE_MISSING = ["latency_min", "dir_5m", "dir_1h", "mag_5m", "mag_1h", "persistence"]


def _price_at(path: pd.Series, t: pd.Timestamp) -> float:
    """Last observed price at or before time t (NaN if none)."""
    sub = path[path.index <= t]
    return float(sub.iloc[-1]) if len(sub) else np.nan


def compute_signature(paths: dict[str, pd.Series], event_time, pre_vol: dict[str, float],
                      checkpoints: list[str], expected: dict | None) -> dict[str, float]:
    """paths[asset] = price series from 60 min before the event to the end of the window.
       Returns a flat dict of signature features.

    checkpoints: subset of ["5m", "1h", "close"]. A path with only 2 points is treated as daily
    (previous close, today's close); otherwise as 1-minute data.
    pre_vol[asset]: std of log returns per path step before the event (per minute or per day).
    expected: {"mag_close": float} for this event type (train mean), or None.
    """
    event_time = pd.Timestamp(event_time)
    gold = paths["gold"].dropna()
    intraday = len(gold) > 2
    sig: dict[str, float] = {"has_intraday": float(intraday)}
    sd = pre_vol["gold"]
    p0 = _price_at(gold, event_time)
    end = gold.index[-1]

    def mag(move: float, minutes: float | None) -> float:
        """Move divided by pre-event std scaled to the horizon (sigma * sqrt(minutes) intraday)."""
        steps = max(minutes, 1.0) if (intraday and minutes is not None) else 1.0
        return move / (sd * np.sqrt(steps)) if sd > 0 else np.nan

    # direction and magnitude at each checkpoint
    for name, minutes in [("5m", 5), ("1h", 60)]:
        if name in checkpoints and intraday:
            mv = np.log(_price_at(gold, event_time + pd.Timedelta(minutes=minutes)) / p0)
            sig[f"dir_{name}"], sig[f"mag_{name}"] = float(np.sign(mv)), mag(mv, minutes)
        else:
            sig[f"dir_{name}"], sig[f"mag_{name}"] = np.nan, np.nan
    mv_close = np.log(float(gold.iloc[-1]) / p0)
    close_min = (end - event_time).total_seconds() / 60
    sig["dir_close"], sig["mag_close"] = float(np.sign(mv_close)), mag(mv_close, close_min)

    if intraday:
        post = gold[gold.index > event_time]
        rel = np.abs(np.log(post / p0))
        hit = rel[rel > 2.0 * sd]  # threshold = latency_threshold_std x pre-event 1-min std
        sig["latency_min"] = float((hit.index[0] - event_time).total_seconds() / 60) if len(hit) else float(close_min)
        post_ret = np.log(post).diff().dropna()
        sig["vol_ratio"] = float(post_ret.std() / sd) if sd > 0 and len(post_ret) > 2 else np.nan
        mv_1h = np.log(_price_at(gold, event_time + pd.Timedelta(minutes=60)) / p0)
        sig["persistence"] = float(np.clip(mv_close / mv_1h, -2, 2)) if mv_1h != 0 else np.nan
    else:
        sig["latency_min"] = np.nan
        sig["vol_ratio"] = abs(mv_close) / sd if sd > 0 else np.nan
        sig["persistence"] = np.nan

    for a in CROSS:
        pa = paths.get(a)
        if pa is None or pa.dropna().empty or not pre_vol.get(a):
            sig[f"xa_{a}_dir"], sig[f"xa_{a}_mag"] = np.nan, np.nan
            continue
        pa = pa.dropna()
        mv = np.log(float(pa.iloc[-1]) / _price_at(pa, event_time))
        steps = max((pa.index[-1] - event_time).total_seconds() / 60, 1.0) if intraday else 1.0
        sig[f"xa_{a}_dir"], sig[f"xa_{a}_mag"] = float(np.sign(mv)), mv / (pre_vol[a] * np.sqrt(steps))

    exp_mag = (expected or {}).get("mag_close", np.nan)
    sig["gap_gold"] = sig["mag_close"] - exp_mag
    sig["dir_mismatch"] = float(np.sign(sig["mag_close"]) != np.sign(exp_mag)) if not np.isnan(exp_mag) else np.nan
    return sig


# ----------------------------------------------------------------------------- daily version
def daily_event_inputs(ev: pd.Series, daily: pd.DataFrame, vol: pd.DataFrame) -> tuple[dict, dict] | None:
    """Two-point price paths (close t-1, close t) and pre-event daily vols (20-day vol at t-1)."""
    t = pd.Timestamp(ev["trading_day"])
    pos = daily.index.get_loc(t)
    if pos < 1:
        return None
    prev = daily.index[pos - 1]
    t_prev = pd.Timestamp(f"{prev.date()} 17:00").tz_localize("America/New_York").tz_convert("UTC")
    t_now = pd.Timestamp(f"{t.date()} 17:00").tz_localize("America/New_York").tz_convert("UTC")
    cols = {"gold": "gold_ret", "usd": "usd_ret", "equities": "eq_ret", "oil": "oil_ret"}
    paths_, pre = {}, {}
    for a, c in cols.items():
        # rebuild a 2-point price path from the log return (start at 1.0)
        paths_[a] = pd.Series([1.0, float(np.exp(daily.loc[t, c]))], index=[t_prev, t_now])
        pre[a] = float(vol.loc[prev, c])
    return paths_, pre


def event_signatures_daily(events: pd.DataFrame, cfg: dict, daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Raw (un-imputed) daily-proxy signature per event, before the gap features are known."""
    if daily is None:
        daily = pd.read_parquet(paths.INTERIM / "daily_ext.parquet")
    vol = daily[["gold_ret", "usd_ret", "eq_ret", "oil_ret"]].rolling(cfg["features"]["roll_vol"]).std()
    rows = []
    for _, ev in events.iterrows():
        inp = daily_event_inputs(ev, daily, vol)
        if inp is None:
            continue
        p, pre = inp
        # event_time for the daily proxy = start of the day-t session (just after the t-1 close)
        et = p["gold"].index[0] + pd.Timedelta(seconds=1)
        rows.append({"event_id": ev["event_id"], **compute_signature(p, et, pre, ["close"], None)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- shared post-processing
def expected_reactions(sig: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> dict:
    """Mean close magnitude of gold and cross assets per event type, over TRAIN events only."""
    df = sig.merge(events[["event_id", "type", "trading_day"]], on="event_id")
    train = df[split_of(df["trading_day"], cfg) == "train"]
    out = {"_fitted_on": {"n_events": int(len(train)), "last_event_day": str(train["trading_day"].max())}}
    for t in TYPES:
        sub = train[train["type"] == t]
        out[t] = {"mag_close": float(sub["mag_close"].mean()),
                  **{f"xa_{a}_mag": float(sub[f"xa_{a}_mag"].mean()) for a in CROSS},
                  "n": int(len(sub))}
    return out


def add_gaps(sig: pd.DataFrame, events: pd.DataFrame, expected: dict) -> pd.DataFrame:
    """gap_gold and dir_mismatch from the train-only expected reactions."""
    df = sig.merge(events[["event_id", "type"]], on="event_id")
    exp = df["type"].map(lambda t: expected[t]["mag_close"])
    df["gap_gold"] = df["mag_close"] - exp
    df["dir_mismatch"] = (np.sign(df["mag_close"]) != np.sign(exp)).astype(float)
    return df.drop(columns="type")


def finalize_event_table(sig: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Impute missing with train mean (0 if all-missing) + *_missing flags; scale by train std."""
    df = sig.merge(events[["event_id", "type", "trading_day", "datetime_utc"]], on="event_id")
    is_train = split_of(df["trading_day"], cfg) == "train"
    stats = {"impute": {}, "scale": {}}
    for c in MAYBE_MISSING:
        miss = df[c].isna()
        df[f"{c}_missing"] = miss.astype(float)
        m = df.loc[is_train, c].mean()
        m = 0.0 if np.isnan(m) else float(m)
        stats["impute"][c] = m
        df[c] = df[c].fillna(m)
    df = df.fillna({c: 0.0 for c in df.columns if c.startswith("xa_")})
    for c in CONT:
        s = df.loc[is_train, c].std()
        s = 1.0 if (np.isnan(s) or s == 0) else float(s)
        stats["scale"][c] = s
        df[c] = df[c] / s
    for t in TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(float)
    return df, stats


def decay_to_days(ev_sig: pd.DataFrame, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Per trading day: s_t = s_e * exp(-lambda * (t - t_e)) using the most recent event e with
    t_e <= t within `decay_max_days` trading days; otherwise 0. If a day has several events, the
    latest one (by release time) is used. Adds days_since_event and last_event_type one-hot."""
    lam, max_d = cfg["reaction"]["decay_lambda"], cfg["reaction"]["decay_max_days"]
    feat = [c for c in ev_sig.columns if c not in ("event_id", "type", "trading_day", "datetime_utc")]
    undecayed = [c for c in feat if c.startswith("type_") or c.endswith("_missing") or c == "has_intraday"]
    decayed = [c for c in feat if c not in undecayed]
    last = ev_sig.sort_values("datetime_utc").groupby("trading_day").tail(1)
    last = last.set_index(pd.to_datetime(last["trading_day"]))
    pos_of = {d: i for i, d in enumerate(days)}
    out = np.zeros((len(days), len(feat) + 1))
    last_ev = [None] * len(days)
    ev_pos = sorted((pos_of[d], d) for d in last.index if d in pos_of)
    j = -1
    for i in range(len(days)):
        while j + 1 < len(ev_pos) and ev_pos[j + 1][0] <= i:
            j += 1
        if j < 0:
            out[i, -1] = max_d
            continue
        k = i - ev_pos[j][0]
        if k > max_d:
            out[i, -1] = max_d
            continue
        row = last.loc[ev_pos[j][1]]
        w = np.exp(-lam * k)
        out[i, :-1] = [row[c] * w if c in decayed else row[c] for c in feat]
        out[i, -1] = k
        last_ev[i] = row["event_id"]
    res = pd.DataFrame(out, index=days, columns=[c if c not in undecayed or not c.startswith("type_")
                                                 else "last_" + c for c in feat] + ["days_since_event"])
    res["days_since_event"] = res["days_since_event"] / max_d  # 0..1
    res.index.name = "date"
    res["last_event_id"] = last_ev
    return res


def build(mode: str, cfg: dict) -> None:
    """Build signatures for `mode` (daily or intraday) and save all outputs."""
    events = pd.read_csv(paths.EVENTS_CSV, parse_dates=["datetime_utc"])
    days = pd.read_parquet(paths.INTERIM / "daily_ext.parquet").index
    if mode == "daily":
        raw = event_signatures_daily(events, cfg)
    else:
        from src.features.intraday_signature import event_signatures_intraday
        raw = event_signatures_intraday(events, cfg)
    raw.to_csv(paths.PROCESSED / f"event_signatures_{mode}_raw.csv", index=False)
    exp = expected_reactions(raw, events, cfg)
    assert exp["_fitted_on"]["last_event_day"] <= cfg["split"]["train_end"]
    if mode == "daily":  # the daily-proxy expectations are the project's expected reactions
        paths.EXPECTED_REACTIONS.write_text(json.dumps(exp, indent=2))
    (paths.PROCESSED / f"expected_reactions_{mode}.json").write_text(json.dumps(exp, indent=2))
    sig = add_gaps(raw, events, exp)
    ev_sig, stats = finalize_event_table(sig, events, cfg)
    stats["expected_fitted_on"] = exp["_fitted_on"]
    (paths.PROCESSED / f"signature_stats_{mode}.json").write_text(json.dumps(stats, indent=2))
    ev_sig.to_csv(paths.PROCESSED / f"event_signatures_{mode}.csv", index=False)
    out = decay_to_days(ev_sig, days, cfg)
    out.to_parquet(paths.SIG_DAILY if mode == "daily" else paths.SIG_INTRADAY)
    print(f"{mode}: {len(ev_sig)} event signatures, {out.shape[1]} day-level columns; "
          f"expected reactions from {exp['_fitted_on']['n_events']} train events "
          f"(last {exp['_fitted_on']['last_event_day']})")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily", "intraday"], default="daily")
    build(ap.parse_args().mode, load_config("base"))


if __name__ == "__main__":
    main()
