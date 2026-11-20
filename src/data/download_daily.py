"""Download daily cross-asset series into data/raw/daily/ (raw files are never modified).

yfinance: DX-Y.NYB (USD index), ^GSPC (S&P 500), BZ=F (Brent), ^VIX, GC=F (gold, fallback only).
FRED:     DGS10 (10Y yield). Uses the FRED API if FRED_API_KEY is set, otherwise FRED's public
          keyless CSV endpoint (same data). See PROGRESS.md decision D0.1.
All these daily closes are published before 17:00 New York, so they are valid for day t.
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from src.utils import paths
from src.utils.config import load_config

YAHOO = {"usd": "DX-Y.NYB", "equities": "^GSPC", "oil": "BZ=F", "vix": "^VIX", "gold_yahoo": "GC=F"}


def download_yahoo(name: str, ticker: str, start: str, end: str) -> None:
    """Save the raw yfinance daily bars for one ticker."""
    out = paths.RAW_DAILY / f"{name}.csv"
    if out.exists():
        return
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    df.to_csv(out)
    print(f"{name} ({ticker}): {len(df)} rows")


def download_fred_series(series_id: str, start: str) -> pd.DataFrame:
    """Fetch a FRED series as a DataFrame [date, value]; API if key available, else public CSV."""
    load_dotenv(paths.ROOT / ".env")
    key = os.getenv("FRED_API_KEY")
    if key:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": series_id, "api_key": key, "file_type": "json",
                                 "observation_start": start}, timeout=60)
        r.raise_for_status()
        obs = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    else:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": series_id, "cosd": start}, timeout=60)
        r.raise_for_status()
        obs = pd.read_csv(io.StringIO(r.text))
        obs.columns = ["date", "value"]
    return obs


def main() -> None:
    """Download everything daily."""
    cfg = load_config("base")
    paths.ensure_dirs()
    start, end = cfg["data"]["download_start"], cfg["data"]["download_end"]
    for name, ticker in YAHOO.items():
        download_yahoo(name, ticker, start, end)
    out = paths.RAW_DAILY / "bonds_DGS10.csv"
    if not out.exists():
        df = download_fred_series("DGS10", start)
        df.to_csv(out, index=False)
        print(f"DGS10: {len(df)} rows")


if __name__ == "__main__":
    main()
