"""Build the event calendar data/interim/events.csv (FOMC, CPI, NFP).

FOMC: scraped from federalreserve.gov (current calendar page + one historical page per year).
      Statement press releases /newsevents/pressreleases/monetaryYYYYMMDDa.htm give the text and
      the release time ("For release at 2:00 p.m. EST"), so unscheduled actions get real times.
CPI / NFP: release dates are the ALFRED vintage dates of CPIAUCSL / PAYEMS. A vintage counts as a
      monthly release only if it adds a new observation month (this removes benchmark/seasonal
      revision vintages). actual = newest value in that vintage, prior = the month before, both
      as published on the release date (first-release values).
Raw HTML/CSV responses are cached in data/raw/events/ so the scrape is reproducible.
"""
from __future__ import annotations

import io
import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from src.utils import paths
from src.utils.config import load_config

FED = "https://www.federalreserve.gov"
# A plain client UA; ALFRED drops connections for some browser-like UA strings.
HEADERS = {"User-Agent": "python-requests (academic research script)"}
NY = "America/New_York"


def fetch_cached(url: str, name: str, params: dict | None = None) -> str:
    """GET `url` once and cache the body in data/raw/events/<name>."""
    out = paths.RAW_EVENTS / name
    r = None
    if out.exists():
        return out.read_text(encoding="utf-8")
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 404:  # ALFRED sometimes returns a transient 404: retry, then give up
                print(f"  404 (attempt {attempt + 1}): {url} {params or ''}")
                time.sleep(3)
                continue
            r.raise_for_status()
            r.encoding = "utf-8"  # both sites serve UTF-8; requests may guess latin-1 wrongly
            out.write_text(r.text, encoding="utf-8")
            time.sleep(0.3)  # be polite to the servers
            return r.text
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}: {url} ({e})")
            time.sleep(3)
    if r is not None and r.status_code == 404:
        return ""
    raise RuntimeError(f"failed to fetch {url}")


# ----------------------------------------------------------------------------- FOMC
def fomc_statement_dates(start_year: int, end_year: int) -> list[str]:
    """All monetaryYYYYMMDDa.htm statement ids linked from the calendar + historical pages."""
    pages = {"fomc_calendar.html": f"{FED}/monetarypolicy/fomccalendars.htm"}
    for y in range(start_year, end_year + 1):
        pages[f"fomc_historical_{y}.html"] = f"{FED}/monetarypolicy/fomchistorical{y}.htm"
    ids = set()
    for name, url in pages.items():
        html = fetch_cached(url, name)
        ids.update(re.findall(r"monetary(\d{8})a\.htm", html))
    return sorted(i for i in ids if start_year <= int(i[:4]) <= end_year)


def parse_release(html: str) -> tuple[str | None, str]:
    """Return (release time 'HH:MM' in ET or None, statement text) from a press-release page."""
    soup = BeautifulSoup(html, "lxml")
    full = soup.get_text(" ", strip=True)
    m = re.search(r"For release at (\d{1,2}):(\d{2}) ([ap])\.m\.", full, re.I)
    hhmm = None
    if m:
        h = int(m.group(1)) % 12 + (12 if m.group(3).lower() == "p" else 0)
        hhmm = f"{h:02d}:{m.group(2)}"
    art = soup.find("div", id="article") or soup
    paras = [p.get_text(" ", strip=True) for p in art.find_all("p")]
    paras = [p for p in paras if p and not p.lower().startswith(("for release", "share", "last update"))]
    text = re.sub(r"\s+", " ", " ".join(paras)).strip()
    return hhmm, text


def build_fomc(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """FOMC policy statements with real release times."""
    start_y, end_y = int(cfg["data"]["start"][:4]), int(cfg["data"]["end"][:4])
    rows, notes = [], []
    for d in fomc_statement_dates(start_y, end_y):
        html = fetch_cached(f"{FED}/newsevents/pressreleases/monetary{d}a.htm", f"fomc_monetary{d}a.html")
        if not html:
            notes.append(f"FOMC {d}: press release page not found")
            continue
        hhmm, text = parse_release(html)
        # Keep only monetary-policy decisions (rate statements), not e.g. longer-run-goals statements.
        if "federal funds rate" not in text.lower():
            notes.append(f"FOMC {d}: excluded (no federal funds rate decision in text)")
            continue
        if hhmm is None:
            hhmm = cfg["events"]["fomc_time_ny"]
            notes.append(f"FOMC {d}: release time not found, used default {hhmm}")
        elif hhmm != cfg["events"]["fomc_time_ny"]:
            notes.append(f"FOMC {d}: non-standard release time {hhmm} ET (from press release)")
        rows.append({"type": "FOMC", "date": d, "time": hhmm, "text": text, "actual": None, "prior": None})
    return pd.DataFrame(rows), notes


# ----------------------------------------------------------------------------- CPI / NFP
def vintage_dates(series: str) -> list[str]:
    """Vintage (release) dates of a FRED series: API if key set, else the ALFRED download page."""
    load_dotenv(paths.ROOT / ".env")
    key = os.getenv("FRED_API_KEY")
    if key:
        r = requests.get("https://api.stlouisfed.org/fred/series/vintagedates",
                         params={"series_id": series, "api_key": key, "file_type": "json"}, timeout=60)
        r.raise_for_status()
        return r.json()["vintage_dates"]
    html = fetch_cached("https://alfred.stlouisfed.org/series/downloaddata", f"alfred_vintages_{series}.html",
                        params={"seid": series})
    soup = BeautifulSoup(html, "lxml")
    sel = soup.find("select", id="form_selected_vintage_dates")
    return sorted({o.get("value") for o in sel.find_all("option") if o.get("value")})


def vintage_values(series: str, vdate: str) -> pd.Series:
    """The whole series as published on `vdate` (ALFRED vintage), monthly index."""
    txt = fetch_cached("https://alfred.stlouisfed.org/graph/alfredgraph.csv",
                       f"alfred_{series}_{vdate}.csv",
                       params={"id": series, "vintage_date": vdate, "cosd": "2010-01-01"})
    if not txt:
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO(txt))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def build_macro(series: str, etype: str, template: str, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Monthly releases for one series (vintages that add a new observation month)."""
    start, end = cfg["data"]["start"], cfg["data"]["end"]
    vds = [v for v in vintage_dates(series) if "2012-10-01" <= v <= end]
    rows, notes, last_obs = [], [], None
    for v in vds:
        s = vintage_values(series, v)
        if len(s) < 2:
            notes.append(f"{etype} {v}: vintage could not be downloaded (persistent 404)")
            continue
        newest = s.index[-1]
        if last_obs is not None and newest <= last_obs:
            notes.append(f"{etype} {v}: revision vintage skipped (no new month)")
            continue
        last_obs = newest
        if v < start:
            continue
        actual, prior = float(s.iloc[-1]), float(s.iloc[-2])
        rows.append({"type": etype, "date": v.replace("-", ""), "time": cfg["events"]["macro_time_ny"],
                     "text": template.format(actual=actual, prior=prior), "actual": actual, "prior": prior,
                     "ref_month": newest.strftime("%Y-%m")})
    return pd.DataFrame(rows), notes


# ----------------------------------------------------------------------------- assemble
def attach_trading_day(ev: pd.DataFrame, trading_days: pd.DatetimeIndex, close_hour: int) -> tuple[pd.DataFrame, list[str]]:
    """Map each event to the trading day whose 17:00 ET close is the first one after the event."""
    notes, tdays = [], []
    for _, r in ev.iterrows():
        ny = r["datetime_ny"]
        d = pd.Timestamp(ny.date())
        if ny.hour >= close_hour:  # at/after the close -> belongs to the next session
            d = d + pd.Timedelta(days=1)
        pos = trading_days.searchsorted(d)
        td = trading_days[pos] if pos < len(trading_days) else pd.NaT
        if pd.notna(td) and td != pd.Timestamp(ny.date()):
            notes.append(f"{r['event_id']}: event at {ny:%Y-%m-%d %H:%M} ET -> trading day {td.date()}")
        tdays.append(td)
    ev["trading_day"] = [t.date().isoformat() if pd.notna(t) else None for t in tdays]
    return ev, notes


def main() -> None:
    """Scrape/download everything and save events.csv."""
    cfg = load_config("base")
    paths.ensure_dirs()
    fomc, n1 = build_fomc(cfg)
    cpi, n2 = build_macro("CPIAUCSL", "CPI", "US CPI release: index {actual} vs prior {prior}", cfg)
    nfp, n3 = build_macro("PAYEMS", "NFP", "US nonfarm payrolls release: {actual} vs prior {prior}", cfg)
    ev = pd.concat([fomc, cpi, nfp], ignore_index=True)
    ev["datetime_ny"] = [pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]} {t}").tz_localize(NY)
                         for d, t in zip(ev["date"], ev["time"])]
    ev["datetime_utc"] = [t.tz_convert("UTC") for t in ev["datetime_ny"]]
    ev["event_id"] = ev["type"] + "_" + ev["date"]
    ev = ev.sort_values("datetime_utc").reset_index(drop=True)
    days = pd.read_parquet(paths.INTERIM / "daily_ext.parquet").index
    ev, n4 = attach_trading_day(ev, pd.DatetimeIndex(days), int(cfg["data"]["close_time_ny"][:2]))
    ev = ev[ev["trading_day"].notna() & (ev["trading_day"] <= cfg["data"]["end"])]
    cols = ["event_id", "type", "datetime_utc", "datetime_ny", "trading_day", "text", "actual", "prior", "ref_month"]
    ev[cols].to_csv(paths.EVENTS_CSV, index=False)
    log = paths.INTERIM / "events_log.txt"
    log.write_text("\n".join(n1 + n2 + n3 + n4))
    print(f"events: {len(ev)} -> {paths.EVENTS_CSV}; notes ({len(n1 + n2 + n3 + n4)}) in {log}")


if __name__ == "__main__":
    main()
