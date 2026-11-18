"""Build the aligned daily table data/interim/daily.parquet.

Master calendar = gold trading days (days with XAU/USD h1 bars before 17:00 New York).
Gold close = price at 17:00 ET = close of the h1 bar that opens at 16:00 ET; if that bar is
missing, the close of the last bar that ends by 17:00 ET that day.
Other series are forward-filled by at most `max_ffill_days` gold trading days.
Target: y_next = 1 if gold_close[t+1] > gold_close[t].

Two files are written:
  daily.parquet      rows in [data.start, data.end] (the modelling period)
  daily_ext.parquet  same columns but including warm-up history from download_start, so
                     rolling features and 30-day windows exist from the first modelling day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import paths
from src.utils.config import load_config

NY = "America/New_York"


def load_gold_hourly() -> pd.DataFrame:
    """Concatenate the yearly XAU/USD h1 CSVs; index = bar open time (UTC)."""
    files = sorted(paths.RAW_INTRADAY.glob("xauusd_h1_*.csv"))
    if not files:
        raise FileNotFoundError("no xauusd_h1_*.csv files; run download_intraday --gold-hourly")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def gold_close_1700(h1: pd.DataFrame, close_hour: int = 17) -> pd.Series:
    """Daily gold close at 17:00 New York from hourly bars (bar end must be <= 17:00 ET)."""
    ny = h1.index.tz_convert(NY)
    bar_end = ny + pd.Timedelta(hours=1)
    # keep bars that END by 17:00 ET of their own NY calendar date (and start that same date)
    end_limit = ny.normalize() + pd.Timedelta(hours=close_hour)
    mask = bar_end <= end_limit
    sub = h1.loc[mask, ["close"]].copy()
    sub["date"] = ny[mask].date
    sub["bar_end_hour"] = bar_end[mask].hour
    last = sub.groupby("date").tail(1).set_index("date")
    close = last["close"].astype(float)
    close.index = pd.to_datetime(close.index)
    close.name = "gold_close"
    n_exact = int((last["bar_end_hour"] == close_hour).sum())
    print(f"gold 17:00 close: {len(close)} days, {n_exact} with an exact 16:00-17:00 bar, "
          f"{len(close) - n_exact} used the last earlier bar")
    return close


def _yahoo_close(name: str) -> pd.Series:
    df = pd.read_csv(paths.RAW_DAILY / f"{name}.csv", index_col=0, parse_dates=True)
    s = pd.to_numeric(df["Close"], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def build_table(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (daily_ext, daily) tables."""
    gold = gold_close_1700(load_gold_hourly())
    gold.to_frame().to_parquet(paths.GOLD_CLOSE_PARQUET)
    cal = gold.index
    lim = cfg["data"]["max_ffill_days"]

    def align(s: pd.Series) -> pd.Series:
        """Reindex to gold trading days, forward-fill at most `lim` days."""
        return s[~s.index.duplicated()].reindex(cal).ffill(limit=lim)

    bonds = pd.read_csv(paths.RAW_DAILY / "bonds_DGS10.csv")
    bonds = pd.Series(pd.to_numeric(bonds["value"], errors="coerce").values,
                      index=pd.to_datetime(bonds["date"])).dropna()
    usd, eq, oil, vix = (align(_yahoo_close(n)) for n in ["usd", "equities", "oil", "vix"])
    y10 = align(bonds)

    df = pd.DataFrame(index=cal)
    df.index.name = "date"
    df["gold_close"] = gold
    df["gold_ret"] = np.log(gold).diff()
    df["usd_ret"] = np.log(usd).diff()
    df["bonds_dchg"] = y10.diff() * 100.0           # yield change in basis points
    df["eq_ret"] = np.log(eq).diff()
    df["oil_ret"] = np.log(oil).diff()
    df["vix"] = vix
    df["vix_chg"] = vix.diff()
    # target: next gold trading day close vs today's close (NaN if t+1 missing)
    nxt = df["gold_close"].shift(-1)
    df["y_next"] = np.where(nxt.isna(), np.nan, (nxt > df["gold_close"]).astype(float))

    n0 = len(df)
    na_by_col = df.isna().sum()
    df_ext = df.dropna()
    print(f"dropped {n0 - len(df_ext)} of {n0} rows with missing values after <= {lim}-day ffill; "
          f"NaNs per column before drop: {na_by_col[na_by_col > 0].to_dict()}")
    start, end = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    df_main = df_ext.loc[start:end].copy()
    df_main["y_next"] = df_main["y_next"].astype(int)
    df_ext = df_ext.copy()
    df_ext["y_next"] = df_ext["y_next"].astype(int)
    return df_ext, df_main


def main() -> None:
    """Build and save the aligned daily tables."""
    cfg = load_config("base")
    paths.ensure_dirs()
    df_ext, df = build_table(cfg)
    df_ext.to_parquet(paths.INTERIM / "daily_ext.parquet")
    df.to_parquet(paths.DAILY_PARQUET)
    print(f"daily.parquet: {len(df)} rows {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()
