# PROGRESS

## Status
Current phase: 1 (daily data)

## Decisions
- D0.1 `.env` exists but is empty (no `FRED_API_KEY`). Instead of blocking, FRED data is fetched from FRED/ALFRED's public keyless CSV endpoints (`fredgraph.csv`, `alfredgraph.csv`, ALFRED vintage list page). Same data as the API. If a key is added, the code uses the API path for vintage dates.
- D0.2 Removed duplicate configs `rc_dgnn_daily.yaml` / `rc_dgnn_intraday.yaml` (superseded by `rcdgnn_daily` / `rcdgnn_full`). Added `baseline_majority`, `baseline_logreg`.
- D0.3 Python 3.11 venv in `.venv` (system default is 3.12; plan asks for 3.11).

## Phase 0: setup ✅
- `pip install -r requirements.txt` OK; torch 2.15.0, transformers 5.23.0, hmmlearn 0.3.3. Lock: `requirements-lock.txt`.
- `npx dukascopy-node --help` OK (Node v25.4.0). Instruments verified with a 1-day h1 test download: `xauusd, usa500idxusd, brentcmdusd, eurusd, dollaridxusd` all exist.
- `pytest` runs (no tests yet). Config loader, seed, paths, Makefile added.
