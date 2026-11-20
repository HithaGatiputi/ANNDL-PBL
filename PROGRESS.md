# PROGRESS

## FINAL SUMMARY (all phases 0–10 complete)
- **Main hypothesis not supported (null result).** Test 2022-01 → 2026-06 (1,159 days), 5 seeds: static_gnn 0.525 ± 0.015 acc, rcdgnn_daily 0.517 ± 0.010, rcdgnn_full 0.516 ± 0.010; best is baseline_lstm 0.538 ± 0.005 vs majority 0.533. No pairwise difference significant (McNemar / Diebold–Mariano p ≥ 0.26), also on post-event days.
- The learned graph barely reacts to events (mean event-driven ΔA ≤ 0.008 on weights ≈ 0.2); removing the L1 penalty does not change validation loss → events carry no usable next-day signal here.
- Data: 3,481 days, 433 events (FOMC 111, CPI 161, NFP 161), 98.4% with valid 1-minute windows. All leakage tests pass (10), 16/16 tests total.
- `make all` re-ran the whole pipeline from raw data: ablation and significance tables reproduced **exactly** (max abs diff 0.0).
- Deliverables: `reports/results/SUMMARY.md`, `ablation_table.md`, `significance.md`, figures in `reports/figures/`, dashboard `app/dashboard.py`, `README.md` (results + limitations), `docs/VIVA_NOTES.md`.
- Sleep blocker (caffeinate) stopped.


## Status
Current phase: done

## Decisions
- D0.1 `.env` exists but is empty (no `FRED_API_KEY`). Instead of blocking, FRED data is fetched from FRED/ALFRED's public keyless CSV endpoints (`fredgraph.csv`, `alfredgraph.csv`, ALFRED vintage list page). Same data as the API. If a key is added, the code uses the API path for vintage dates. **Update:** key added later; the API's vintage dates were cross-checked against the scraped ALFRED list for 2012-10 → 2026-06: CPIAUCSL 174 = 174, PAYEMS 165 = 165, no differences.
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

## Phase 2: events and regimes ✅
- **433 events** (FOMC 111, CPI 161, NFP 161). Per year: FOMC 8 (2019: 9, 2020: 10 incl. unscheduled), CPI/NFP 12 per year; 2025 has 11 each (Oct-2025 government shutdown: October CPI cancelled, Oct+Nov jobs combined); 2026 = Jan–Jun only.
- Unscheduled FOMC actions with real times from the press releases: 2019-10-11 11:00, 2020-03-03 10:00, 2020-03-15 17:00 (Sunday → trading day 2020-03-16), 2020-03-23 08:00. Excluded (no rate decision in text): 2020-03-31, 2020-08-27, 2025-08-22.
- 2013–2015 press releases print no release time → default 14:00 ET, **verified from data**: on those 24 FOMC days mean |h1 gold return| is 67 bp in the 14:00–15:00 bar vs 10–16 bp in the 11:00–14:00 bars (12:30 releases would spike the 12:00 bar).
- CPI/NFP: 10 CPI seasonal-revision vintages (Feb) and 1 NFP vintage removed by the "must add a new month" rule. 7 events on non-trading days moved to the next trading day (logged in `data/interim/events_log.txt`). 10 trading days carry 2 events.
- Spot-checks (vs official pages): FOMC 2020-03-15 17:00 ET cut to 0–1/4% ✓; FOMC 2022-06-15 14:00 to 1½–1¾% ✓; FOMC 2013-06-19 14:00 held 0–1/4% ✓; CPI 2022-07-13 June index 295.328 vs 291.474 ✓; NFP 2020-05-08 131,072k vs 151,572k (−20.5M) ✓.
- Decision D2.1: HMM labels are decoded with the forward (filtering) recursion, not Viterbi/forward-backward, so day-t labels never use later data. Inputs standardised with train mean/std.
- Decision D2.2: ALFRED returns transient 404s and drops browser-like user agents → plain UA + retry; pages forced to UTF-8.
- Regimes (2013–2026-06): vol-percentile calm 786 / normal 1245 / volatile 1145 / crisis 305; HMM calm 1151 / normal 1431 / volatile 899. **Cohen's κ = 0.602**. March 2020: 100% volatile/crisis in both. Plot: `reports/figures/phase2_regimes.png`.

## Phase 3: baselines + leakage tests ✅
- Dataset: X (30, 5 nodes, 3 feats), global G (RSI, MACD hist/price, VIX, ΔVIX, regime one-hot). StandardScaler fit on train rows only. Samples: train 1,805 / val 517 / test 1,159.
- `tests/test_leakage.py`: **7/7 pass** — future-deletion (3 cut dates, X/S/E/G identical), regime+HMM causality, train-only fitting (perturbing val/test data leaves train features, thresholds and HMM unchanged), split target periods ordered, shuffled-label LSTM test acc 0.517/0.527 vs majority 0.533.
- Majority/logreg are deterministic → 1 run (std 0). LSTM: 4,769 params, 5 seeds.

| variant | subset | n | acc | MCC | Brier |
|---|---|---|---|---|---|
| baseline_majority | all | 1159 | 0.533 | 0.000 | 0.250 |
| baseline_logreg | all | 1159 | 0.517 | 0.033 | 0.249 |
| baseline_lstm | all | 1159 | 0.538 ± 0.005 | 0.036 ± 0.024 | 0.249 |
| baseline_lstm | post_event | 141 | 0.489 ± 0.009 | 0.076 ± 0.052 | 0.251 |

No baseline > 60% (no leakage alarm). LSTM early-stops at epochs 0–5: there is very little learnable signal.

## Phase 4: reaction signature (daily) ✅
- `compute_signature` handles 1-min and daily paths. 433 event signatures → `signatures_daily.parquet` (29 cols incl. `days_since_event`, `last_type_*`, `*_missing`).
- Expected reactions from **225 train events only** (last 2019-12-11; asserted). FOMC mag_close +0.21, CPI +0.20, NFP +0.11 (daily-σ units).
- Decisions: D4.1 magnitudes = move ÷ (pre-event σ·√steps) so 1-min and daily magnitudes share a unit (plan says "move ÷ pre-event std"; the √time keeps intraday/daily-fallback rows comparable). D4.2 continuous signature features are divided by their train-event std (no centering, so 0 still means "no event"). D4.3 on days with 2 events the later release is used for the decayed day signature. D4.4 daily proxy uses the 20-day vol at t−1 as "pre-event" vol.
- Known events (daily proxy, close t−1→t): 2013-06-19 gold −1.11σ, USD +2.09σ, equities −1.61σ ✓ (taper tantrum). 2020-03-15 (→ 3/16) gold −0.57σ, equities −2.89σ, USD −0.96σ. **2022-06-15: gold +1.45σ, USD −0.60σ** — the plan expected USD up; the data (DXY close 105.52 → 105.16) shows USD *fell* close-to-close that day. Reported as-is; the intraday check is in Phase 7.
- Tests: synthetic-path test (latency 3 min, magnitudes, persistence 0.5) ✓; new leakage test: event signature & decay causal, expected reactions train-only ✓. `pytest`: 11 passed.

## Phase 5: FinBERT embeddings + news baseline ✅
- ProsusAI/finbert frozen; statements chunked ≤512 tokens, mean-pooled [CLS]. PCA 768→16 fit on 225 train events (explains 95.5%), components ÷ train std. `event_emb.parquet`: 433 rows, no NaNs (test added). Decayed daily version: 16 emb + 3 sentiment.
- Mean sentiment: CPI neg 0.69 (template text "index X vs prior Y" reads as negative to FinBERT — the CPI/NFP texts carry little semantic content; noted as a limitation), FOMC neutral 0.70, NFP neutral 0.59.
- `baseline_lstm_news` (4,788 params): test acc **0.536 ± 0.011**, MCC 0.044 (vs baseline_lstm 0.538 ± 0.005).
- Intraday 1-min download for Phase 7 started in the background (~25 files/min).

## Phase 6: static GNN and RC-DGNN (daily) ✅ — null result
- Architecture as specified (5-node graph in plain PyTorch, A = rowsoftmax(A_static + gate·ΔA + GAT), 2 residual graph layers, LSTM, MLP head, MC-dropout T=30). Tests: forward shapes + rows of A sum to 1 ✓, static variant ignores the signature ✓, **overfit test: 100% train acc on 64 samples** ✓.
- D6.1 Both GNNs feed the day-t event embedding to the head, so the *only* difference static vs RC is the reaction-conditioned edges. L1 is applied to the effective change gate·ΔA.
- D6.2 **Overfitting fix (plan §15)**: with the specified sizes (32/64, ~40k params) val loss rose from epoch 0–1 while train loss fell (`reports/results/_tuning/*_loss.png`). Switched both GNNs to hidden 16, dropout 0.4 (graph 0.3), L1 0.01. Selection on **validation only**: mean val log loss 0.6929 → 0.6924. Test metrics of tuning runs were not used. Params now: static 4,842, RC-DGNN 6,083.

**Key comparison (test, 5 seeds):**

| variant | all: acc | all: MCC | all: Brier | post-event (n=141): acc | post-event: MCC |
|---|---|---|---|---|---|
| baseline_lstm | 0.538 ± 0.005 | 0.036 | 0.249 | 0.489 ± 0.009 | 0.076 |
| baseline_lstm_news | 0.535 ± 0.011 | 0.044 | 0.249 | 0.495 ± 0.020 | 0.034 |
| static_gnn | 0.525 ± 0.015 | 0.014 | 0.249 | 0.492 ± 0.057 | 0.031 |
| rcdgnn_daily | 0.517 ± 0.010 | 0.017 | 0.250 | 0.492 ± 0.029 | 0.013 |

RC-DGNN (daily signature) does **not** beat the static GNN on all days or on post-event days; all neural models are at or below the majority rate (0.533). Significance tests follow in Phase 8.

## Phase 7: intraday signatures ✅
- 1,732 m1 windows downloaded (433 events × gold/USD/US500/Brent), 0 failures; resumable. `dollaridxusd` has no Dukascopy data for most 2013–2017 days → EUR/USD (sign-inverted) fallback downloaded for 179 events (D7.1). USD source: dollar index 254, EUR/USD 172, daily proxy 7.
- **Coverage (valid gold window): 98.4%** (CPI 98.8%, FOMC 99.1%, NFP 97.5%); per type × year in `scripts/phase7_checks.py` output (lowest cell NFP 2026 83% = 1 of 6). The 7 fallbacks include FOMC 2020-03-15 (Sunday 17:00, market shut → daily proxy, `has_intraday=0`). Cross-asset valid rate: USD 98.4%, equities 84.8%, oil 93.5% (poor windows keep the daily-proxy value). Bonds stay daily (limitation).
- Intraday 2022-06-15 FOMC: gold fell in the first 5 min, rose by +1h and the close; USD (DXY m1) fell by 17:00 (−1.86σ) — consistent with the daily data, not with the plan's expectation of a USD rise.
- Figure: `reports/figures/phase7_event_paths.png`. New leakage test: all intraday bars lie within [event−60 min, 17:00 ET of day t] ✓. `pytest`: 16 passed.
- Latency (plan definition: |move| > 2 × pre-event 1-min σ) is ~1 minute for most events, so it carries little information (noted).

**Second key comparison (test, 5 seeds):** rcdgnn_daily 0.517 ± 0.010 (MCC 0.017) vs **rcdgnn_full 0.516 ± 0.010 (MCC 0.016)**; post-event 0.492 vs 0.489. Intraday reaction data did not help.

## Phase 8: evaluation, significance, explainability ✅ (`make evaluate`, one command)
- Outputs: `ablation_table.{csv,md}` (7 configs × 6 subsets × 6 metrics, + val), `significance.{csv,md}`, `event_analysis.md`, `backtest.md`, `SUMMARY.md`, `graph_change_by_type.csv`, `permutation_importance.csv`; figures `calibration.png`, `graph_change_<event>.png` (FOMC 2020-03-15, FOMC 2022-06-15, CPI 2022-11-10, NFP 2023-02-03), `graph_change_by_type.png`, `permutation_importance.png`.
- **Significance (test, seed-ensembled): nothing is significant.** static vs rcdgnn_daily: acc 0.528 vs 0.528, McNemar p=1.00, DM p=0.39; post-event 0.461 vs 0.511, McNemar p=0.26. rcdgnn_daily vs full: p=1.00 / DM 0.30. baseline_lstm vs rcdgnn_full: 0.532 vs 0.529, p=0.86 / DM 0.47.
- Why daily ≈ full: the signature files differ, but predictions differ by < 0.001 because the learned event edge change is tiny (largest mean test-event ΔA −0.0077 on weights ≈ 0.2; permutation importance of every signature feature ≈ 0; only the event embedding in the head has any (tiny) effect).
- D8.1 Diagnostic (labelled, not used to choose the model): L1 = 0 → ΔA ~5× larger (event-day mean |ΔA| 0.012, max 0.11) but val log loss unchanged (0.6927/0.6928), test acc 0.515. The null result is **not** caused by the penalty. Runs kept in `reports/results/_tuning/`.
- Backtest (simplified, not evidence of profit): no model beats buy-and-hold (Sharpe 0.96); RC-DGNN Sharpe −0.53.

## Phase 9: dashboard ✅
- `streamlit run app/dashboard.py` (or `make dashboard`): Forecast (P(up), MC band, regime, last event + days since, outcome), Graph (5-node graph day before / event day / selected day + event-driven ΔA table), Event explorer (1-min gold path with checkpoints, signature, expected vs actual), Results (SUMMARY, tables, calibration, graph and importance figures). Reads saved files only.
- Verified with Streamlit `AppTest`: all 4 pages × 2 models run with 0 exceptions, 0.2–1.2 s per page (< 5 s).

## Phase 10: documentation and cleanup ✅
- README updated (architecture as built, exact sources/dates, results table, findings, limitations incl. consensus-forecast and bond-intraday gaps). Docstrings added to every public function (AST check: 0 missing). `docs/VIVA_NOTES.md` written from actual results.
- `make all` from raw data: data → features → 7 configs → evaluate → tests; outputs identical to the earlier run; `pytest` 16 passed.

## Audit (2026-11-20) ✅
- **Leakage re-checked:** no event after its day-t 17:00 ET cutoff; no hidden NaN/inf in processed features; extreme returns are real (gold −9.6% 2013-04-15, Brent −28% Mar-2020).
- **Suspicious logreg backtest (66% hit rate on 150 long days) investigated — not leakage.** Truncation tests pass; gold returns have no lag-1 autocorrelation (train −0.009, test 0.006), so no stale-close artifact; 105 of 150 long days fall in the 2025–26 rally and the same rule shows no edge on validation (56.1% vs 55.1% base). Rally coincidence, not skill.
- **Fixed: config values silently ignored.** `compute_signature` hard-coded latency 2σ, checkpoints 5/60 min and persistence clip ±2 while `base.yaml` had `latency_threshold_std`, `checkpoints_min`, `persistence_clip`. Now read from config. Also moved to config: min pre-event bars, download end date, stress periods, permutation repeats, test-start date (graph_viz, dashboard), vintage scan start, close hour in `align`. Stale `rc_dgnn` docstring fixed. New test: changing the config changes the signature.
- Rebuilt data/events/signatures: byte-identical to before → model results unchanged; `make evaluate` outputs unchanged; `pytest` 17 passed.
- Note (not a leak): FRED DGS10 for day t is the ~15:30 ET market yield, so it is valid at 17:00, but the official number is published after 17:00 — a live system would need a real-time yield feed.
