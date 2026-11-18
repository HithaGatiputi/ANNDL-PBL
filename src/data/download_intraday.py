"""Download Dukascopy bars with the `dukascopy-node` CLI (called through subprocess).

Two jobs:
  --gold-hourly : XAU/USD h1 bars in yearly chunks -> data/raw/intraday/xauusd_h1_<year>.csv
  --events      : 1-minute windows around every event -> data/raw/intraday/<event_id>_<asset>.csv

Both are resumable: a file that already exists is skipped, failures are retried 3 times.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

from src.utils import paths
from src.utils.config import load_config

# Verified in Phase 0 with a 1-day test download.
INSTRUMENTS = {"gold": "xauusd", "usd": "dollaridxusd", "equities": "usa500idxusd", "oil": "brentcmdusd"}
RETRIES = 3


def dukascopy_download(instrument: str, date_from: str, date_to: str, timeframe: str, out_file: Path) -> bool:
    """Download one chunk to `out_file` (CSV with timestamp in UTC ms). Returns True on success.

    `date_to` is exclusive in dukascopy-node. Bars with zero volume ("flats") are excluded,
    so market-closed hours do not appear as fake prices.
    """
    for attempt in range(1, RETRIES + 1):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = ["npx", "-y", "dukascopy-node", "-i", instrument, "-from", date_from, "-to", date_to,
                   "-t", timeframe, "-f", "csv", "-v", "-dir", tmp, "-s", "-r", "2"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            files = list(Path(tmp).glob("*.csv"))
            if res.returncode == 0 and files:
                shutil.move(str(files[0]), out_file)
                return True
            # An empty range (e.g. market closed) produces no file: treat as success with empty CSV.
            if res.returncode == 0 and not files:
                pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]).to_csv(out_file, index=False)
                return True
        print(f"  retry {attempt}/{RETRIES} for {instrument} {date_from}: {res.stderr[-200:]}")
        time.sleep(2 * attempt)
    return False


def download_gold_hourly(cfg: dict) -> None:
    """Yearly chunks of XAU/USD hourly bars."""
    start_year = int(cfg["data"]["download_start"][:4])
    end_year = int(cfg["data"]["end"][:4])
    for year in range(start_year, end_year + 1):
        out = paths.RAW_INTRADAY / f"xauusd_h1_{year}.csv"
        if out.exists():
            continue
        # one extra week so the last trading day of the period has t+1
        to = f"{year + 1}-01-01" if year < end_year else "2026-07-08"
        ok = dukascopy_download("xauusd", f"{year}-01-01", to, "h1", out)
        print(f"xauusd h1 {year}: {'ok' if ok else 'FAILED'}")


def event_window_utc(ev: pd.Series, pre_min: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Reaction window: from `pre_min` minutes before the event to 17:00 ET on its trading day."""
    start = pd.Timestamp(ev["datetime_utc"]) - pd.Timedelta(minutes=pre_min)
    end_ny = pd.Timestamp(f"{ev['trading_day']} 17:00").tz_localize("America/New_York")
    return start, end_ny.tz_convert("UTC")


def download_event_windows(cfg: dict) -> None:
    """1-minute bars for each event and asset; the day-range download is cut to the window later."""
    events = pd.read_csv(paths.EVENTS_CSV)
    pre = cfg["reaction"]["pre_event_min"]
    failed = []
    for i, ev in events.iterrows():
        start, end = event_window_utc(ev, pre)
        for asset, inst in INSTRUMENTS.items():
            out = paths.RAW_INTRADAY / f"{ev['event_id']}_{asset}.csv"
            if out.exists():
                continue
            d_from = start.strftime("%Y-%m-%d")
            d_to = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if not dukascopy_download(inst, d_from, d_to, "m1", out):
                failed.append(f"{ev['event_id']}_{asset}")
        if i % 25 == 0:
            print(f"events done: {i + 1}/{len(events)}", flush=True)
    print(f"event windows finished; failed: {len(failed)} {failed[:10]}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-hourly", action="store_true")
    ap.add_argument("--events", action="store_true")
    args = ap.parse_args()
    cfg = load_config("base")
    paths.ensure_dirs()
    if args.gold_hourly:
        download_gold_hourly(cfg)
    if args.events:
        download_event_windows(cfg)


if __name__ == "__main__":
    main()
