# PROGRESS

## Status
Current phase: 2 (events, regimes)

## Decisions
- D0.1 `.env` exists but is empty (no `FRED_API_KEY`). Instead of blocking, FRED data is fetched from FRED/ALFRED's public keyless CSV endpoints (`fredgraph.csv`, `alfredgraph.csv`, ALFRED vintage list page). Same data as the API. If a key is added, the code uses the API path for vintage dates.
- D0.2 Removed duplicate configs `rc_dgnn_daily.yaml` / `rc_dgnn_intraday.yaml` (superseded by `rcdgnn_daily` / `rcdgnn_full`). Added `baseline_majority`, `baseline_logreg`.
- D0.3 Python 3.11 venv in `.venv` (system default is 3.12; plan asks for 3.11).

## Phase 0: setup ✅
- `pip install -r requirements.txt` OK; torch 2.15.0, transformers 5.23.0, hmmlearn 0.3.3. Lock: `requirements-lock.txt`.
- `npx dukascopy-node --help` OK (Node v25.4.0). Instruments verified with a 1-day h1 test download: `xauusd, usa500idxusd, brentcmdusd, eurusd, dollaridxusd` all exist.
- `pytest` runs (no tests yet). Config loader, seed, paths, Makefile added.

## Phase 1: daily data ✅
- Gold = Dukascopy XAU/USD h1 → 17:00 ET close (3,635 days exact 16–17h bar; 110 early-close days used last earlier bar; no day had <12 bars). No GC=F fallback needed.
- `daily.parquet`: **3,481 rows**, 2013-01-02 → 2026-06-30, **0 NaNs**. Inside the period no rows were dropped (only the warm-up first row 2012-06-01 and the final row without t+1). `daily_ext.parquet` keeps 2012-06 warm-up history for rolling features/windows.
- `y_next` up-rate: train 50.6% (1,805), val 55.1% (517), test 53.3% (1,159).
- corr(gold_ret, usd_ret) = **−0.416** (negative ✅). Plot: `reports/figures/phase1_prices.png`.
