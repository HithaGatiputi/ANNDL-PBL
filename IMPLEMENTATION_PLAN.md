# IMPLEMENTATION_PLAN.md: RC-DGNN for Event-Aware Gold Forecasting

This file is the build spec for the project. `README.md` gives the overview; this file says **exactly what to build, in what order, and how to know each step is done.**

---

## 0. Instructions for Claude Code (read first)

0. **Run autonomously, start to finish.** The user is away. Do not ask questions or wait for approval. Work through Phases 0–10 in order without stopping. When a decision is needed, pick the most reasonable option that follows this plan, log it in `PROGRESS.md` under "Decisions", and keep going.
   - **Keep the computer awake for the whole run.** As your very first command, start a background sleep blocker:
     - macOS: `caffeinate -dimsu &` (save its PID to `.caffeinate.pid`)
     - Windows (PowerShell): `powercfg /change standby-timeout-ac 0; powercfg /change monitor-timeout-ac 0` (note the original values in `PROGRESS.md` and restore them at the end)
     - Linux: `systemd-inhibit --what=idle:sleep sleep infinity &`
   - **Run long jobs (downloads, training) in the background**, writing logs to `logs/<step>.log`, and check on them rather than blocking. Downloads must be resumable, so a crash or restart loses no work.
   - **`PROGRESS.md`** is the user's view of the run. Keep it updated with the current phase, what's done, the check results, decisions, problems and the results tables.
   - **When everything is done**, stop the sleep blocker (or restore the Windows power settings), and write a final summary at the top of `PROGRESS.md`.
1. **Work one phase at a time, in order.** At the end of each phase, run its acceptance checks, write the report to `PROGRESS.md`, commit, and move straight on to the next phase. If a check fails, fix it before moving on. If it can't be fixed after a reasonable effort, log it clearly and continue with whatever doesn't depend on it.
2. **Never fabricate data or results.** If a download fails, a source changed, or a number looks wrong, use the fallback in this plan (section 15) and log it in `PROGRESS.md`. Do not fill gaps with synthetic data, except in unit tests, and label those clearly.
3. **Leakage is the #1 risk.** Every feature for day *t* may use only information available by **17:00 New York time on day *t***. The target is gold's move from day *t* close to day *t+1* close. Section 3 has the rules, and `tests/test_leakage.py` must pass before any model results are reported.
4. **Fit on train only.** Scalers, expected reactions, regime thresholds, the HMM and anything else learned from data are fitted on the training split only, then applied to val and test.
5. **Keep it small.** About 3,400 trading days and about 400 events. Models should have fewer than 100k parameters and train on a laptop CPU in minutes.
6. **Prefer simple, readable code.** These are students who must explain every line in a viva. Write docstrings, type hints and short comments explaining *why*. No clever abstractions.
7. **Everything runs from configs.** Put no magic numbers in code. Read them from `configs/base.yaml` plus the variant config.
8. **Commit after each phase**, with a clear message, if the repo is a git repo.
9. **If anything in this plan is wrong or impossible** (an API changed, an instrument is unavailable), apply the smallest fix that keeps the design, log it in `PROGRESS.md` under "Decisions", and continue.

---

## 1. Project summary

**RC-DGNN (Reaction-Conditioned Dynamic Graph Neural Network)** predicts **P(gold closes higher tomorrow)**.

Core idea:
1. When a scheduled macro event happens (FOMC decision, CPI release, US jobs report / NFP), measure the **early market reaction** from 1-minute data: latency, direction, magnitude, volatility change, persistence, cross-asset response, and the gap between expected and actual reaction.
2. Turn that into a **Market Reaction Signature** vector.
3. Use the signature (plus a FinBERT embedding of the event text) to **generate edge weights** for a 5-node asset graph: Gold, USD, Bonds, Equities, Oil.
4. Run graph layers on each day of a 30-day window, then a temporal model (LSTM), then output a probability with uncertainty.

**Main hypothesis to test:** the reaction-conditioned graph beats an otherwise identical **static** graph, especially on days right after events.

---

## 2. Environment and setup (Phase 0)

### 2.1 Stack
- Python 3.11, with a virtual environment in `.venv`
- `numpy pandas pyarrow scikit-learn torch transformers hmmlearn shap matplotlib pyyaml streamlit pytest requests beautifulsoup4 lxml yfinance python-dotenv statsmodels`
- **Do not use torch-geometric.** With only 5 nodes, implement graph attention directly in PyTorch, which is simpler and installs everywhere. Remove `torch-geometric` and `dukascopy-python` from `requirements.txt`.
- Intraday data: **Node.js + `dukascopy-node`** via `npx dukascopy-node ...`, called from Python with `subprocess`. Check Node is installed. If it isn't, tell the user to install Node LTS.

### 2.2 Secrets
- `.env` holds `FRED_API_KEY=...`, a free key from fred.stlouisfed.org. Load it with `python-dotenv`, and never commit `.env`.
- The user puts the key in `.env` before starting. If it is missing, log it in `PROGRESS.md`, build every part that does not need FRED, and come back to the FRED parts at the end. Never hard-code the key.

### 2.3 Repo changes in Phase 0
- Update `requirements.txt` (see 2.1) and pin versions after install with `pip freeze > requirements-lock.txt`.
- Change the start date in `configs/base.yaml` to **2013-01-01**. FOMC statement times were inconsistent in 2012: some came out at 12:30 PM ET. Update the README period line to match.
- Implement the config loader `src/utils/config.py`: load `configs/<variant>.yaml`, merge it over `base.yaml`, and return a dict.
- Add `src/utils/seed.py` (seeds numpy, torch and random) and `src/utils/paths.py` (all data paths in one place).
- Add a `Makefile`, or extend `scripts/run_pipeline.sh`, with the targets `data`, `features`, `train VARIANT=...`, `ablation`, `evaluate`, `dashboard` and `test`.

### 2.4 Acceptance checks
- `pip install -r requirements.txt` succeeds, and `python -c "import torch, transformers, hmmlearn"` works.
- `npx dukascopy-node --help` prints its help text.
- `pytest` runs, even if there are no tests yet.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 3. Time conventions and leakage rules (apply everywhere)

- **All timestamps** are stored as timezone-aware UTC and converted to `America/New_York` for day logic.
- **Trading day *t*** = the New York calendar date whose session closes at **17:00 ET**.
- **Gold daily close (the target series)** = the XAU/USD price at **17:00 ET**, taken from **Dukascopy hourly (h1) bars**. This keeps the target perfectly aligned with the reaction window.
  - *Fallback,* only if Dukascopy fails: Yahoo `GC=F` daily close, with the reaction window cut off at **13:30 ET**. Report it if you use this fallback.
- **Target:** `y_t = 1 if gold_close_{t+1} > gold_close_t else 0`. Drop days where t+1 is missing.
- **Features for day *t*** use only data timestamped ≤ 17:00 ET on day *t*.
- **Reaction window:** from 60 minutes before the event to **17:00 ET on the event day**. The target starts at that same 17:00 ET close, so there is no overlap.
- **News/text:** only text published before 17:00 ET on day *t* is attached to day *t*.
- **Splits (fixed, time-ordered):** train 2013-01-01 → 2019-12-31, val 2020-01-01 → 2021-12-31, test 2022-01-01 → 2026-06-30. Never shuffle across splits. Shuffling batches *within* the train split is fine.
- **Window rule:** a 30-day input window for day *t* covers days t-29 … t. The first 29 days of val and test may reach back into the previous split for **inputs only**, never for targets. That is allowed and is not leakage.

---

## 4. Phase 1: Daily market data

**Files:** `src/data/download_daily.py`, `src/data/download_intraday.py` (the hourly gold part only), `src/data/align.py`

### 4.1 Sources
| Node | Series | Source | Transform |
|---|---|---|---|
| Gold | XAU/USD h1 → 17:00 ET close | Dukascopy (`xauusd`, timeframe `h1`) | log return |
| USD | US Dollar Index `DX-Y.NYB` | yfinance | log return |
| Bonds | US 10Y yield `DGS10` | FRED API | **daily change in yield (bp)**, not a return |
| Equities | S&P 500 `^GSPC` | yfinance | log return |
| Oil | Brent `BZ=F` | yfinance | log return |
| (extra feature) | VIX `^VIX` | yfinance | level and change |

- **Use Brent, not WTI (`CL=F`).** WTI went negative in April 2020, which breaks log returns.
- Download raw files into `data/raw/daily/` (CSV or parquet), keeping the raw data untouched.
- Dukascopy hourly gold: download in yearly chunks to `data/raw/intraday/xauusd_h1_<year>.csv`, then build the 17:00 ET close. If the 17:00 bar is missing (for example on a holiday), take the last bar before 17:00 that day. If there's no bar on that day at all, it isn't a trading day.
- Before downloading, check the instrument IDs with `npx dukascopy-node` (its docs and instrument list). The IDs used in this plan, `xauusd`, `usa500idxusd`, `brentcmdusd`, `eurusd` and `dollaridxusd`, are **expected** to exist but must be verified. Report any that don't exist.

### 4.2 Alignment (`align.py`)
- The master calendar is the gold trading days.
- Forward-fill other series by **at most 2 days**, for holidays that differ between markets. Drop rows still missing values and **log how many were dropped and why**.
- Output: `data/interim/daily.parquet`, indexed by date, with columns `gold_close, gold_ret, usd_ret, bonds_dchg, eq_ret, oil_ret, vix, vix_chg` and the target `y_next`.

### 4.3 Acceptance checks
- Roughly 3,300–3,500 rows from 2013 to 2026-06, with no NaNs in the final table.
- Print the class balance of `y_next` per split. Expect about 50–55% up days, and note that the 2024–25 gold rally makes the test split lean "up".
- Sanity plot `reports/figures/phase1_prices.png`, with gold and the four other series normalised.
- Correlation check: gold vs USD return correlation should be **negative** over the full period. Report the value.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 5. Phase 2: Event calendar and regime labels

**Files:** `src/data/download_events.py`, `src/features/regime.py`

### 5.1 Events (`data/interim/events.csv`)
Columns: `event_id, type, datetime_utc, datetime_ny, trading_day, text, actual, prior`

| Type | Dates from | Time (ET) | Text |
|---|---|---|---|
| **FOMC** | Scrape federalreserve.gov: the FOMC calendar pages, plus historical pages per year | Statement time, **14:00** from 2013 onward (verify) | Scrape the statement press release text. URLs look like `/newsevents/pressreleases/monetaryYYYYMMDDa.htm` |
| **CPI** | FRED API `series/vintagedates` for `CPIAUCSL` (release dates) | **08:30** | Template: `"US CPI release: index {actual} vs prior {prior}"` |
| **NFP** | FRED API `series/vintagedates` for `PAYEMS` | **08:30** | Template: `"US nonfarm payrolls release: {actual} vs prior {prior}"` |

- Include **unscheduled FOMC actions**, such as March 3 and March 15, 2020, with their real times. Verify the times from the press releases.
- Ignore vintage dates that are annual benchmark revisions, not monthly releases. Expect about 12 dates per year for each of CPI and NFP. Report and inspect any year that has a different count.
- `actual` and `prior` come from the first-release values via the FRED API (`realtime_start` = release date).
  - **Limitation to note:** economist consensus forecasts aren't freely available, so the "surprise" is the actual value vs. the prior month's value, not vs. consensus.
- If an event falls on a non-trading day, attach it to the next trading day and log it.
- Save the raw scraped HTML into `data/raw/events/` so the scraping is reproducible.

### 5.2 Regime labels (`regime.py`)
- 20-day rolling standard deviation of `gold_ret`.
- Thresholds from **train-split** percentiles: < 33rd percentile = calm, 33–67 = normal, > 67 = volatile, > 95th = crisis (crisis overrides volatile).
- HMM check: `hmmlearn.hmm.GaussianHMM(n_components=3)`, fitted on train `[gold_ret, rolling_vol]` and applied to all data. Map the states to calm, normal and volatile by their mean volatility.
- Output: `data/interim/regimes.parquet` with columns `regime_vol, regime_hmm`.

### 5.3 Acceptance checks
- About 32 events per year (about 8 FOMC, 12 CPI, 12 NFP) → roughly 430 events in total. Print the counts per year and type.
- Spot-check 5 events by hand against the official pages, and print them for the user.
- Regimes: both methods label most of **March 2020** volatile or crisis. Print the agreement rate between them (Cohen's kappa) and a plot `reports/figures/phase2_regimes.png`.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 6. Phase 3: Baselines

**Files:** `src/training/splits.py`, `src/training/dataset.py`, `src/models/baselines.py`, `src/training/train.py`, `src/evaluation/metrics.py`, `tests/test_leakage.py`

### 6.1 Dataset
- `dataset.py` builds, for each day *t*, a tensor `X_nodes[t]` of shape `(L=30, N=5, F)`. The per-node features for each day τ in the window are:
  - the return (or yield change)
  - the 5-day rolling return
  - the 20-day rolling volatility
- Global features for day *t*: gold RSI(14), MACD histogram, VIX level, VIX change, and the regime one-hot.
- Scale with `StandardScaler`, fitted on train only.
- Each sample also carries `date`, `y`, `is_event_day`, `is_post_event_day` (the day after an event) and `regime`.

### 6.2 Baselines (configs already exist; add any that are missing)
| Config | Model |
|---|---|
| `baseline_majority` | Always predict the train-split majority class |
| `baseline_logreg` | Logistic regression on gold's last 5 returns plus the global features |
| `baseline_lstm` | LSTM on the gold node's features only (hidden 32, 1 layer, dropout 0.3) |

### 6.3 Training loop (`train.py`), shared by all neural models
- Loss: `BCEWithLogitsLoss`. Optimizer: AdamW (lr 1e-3, weight_decay 1e-4). Batch size 64. Up to 50 epochs.
- **Early stopping on validation log loss** with patience 8. Keep the best checkpoint.
- Run 5 seeds `[0,1,2,3,4]` and save every seed's predictions.
- Save `reports/results/<variant>/seed<k>/preds_{val,test}.csv` with the columns `date, y, p_up, p_std`, plus `metrics.json`.
- Log the train/val loss curves to `reports/figures/<variant>_loss.png`.

### 6.4 Metrics (`metrics.py`)
- Accuracy, F1, MCC, Brier score, log loss, ECE (10 bins)
- Computed on: all test days; event days; post-event days; and each regime
- Aggregate as the mean ± std across seeds

### 6.5 Leakage tests (`tests/test_leakage.py`), required
1. Every feature row for day *t* uses only data with timestamps ≤ 17:00 ET on day *t*. Test this by building features with the future rows deleted and asserting they are identical.
2. The scaler, the regime thresholds and the HMM were fitted only on train dates.
3. There is no date overlap between the target periods of the train, val and test splits.
4. **Shuffled-label test:** train `baseline_lstm` with randomly permuted train labels. Test accuracy must be close to the majority-class rate. If it's much higher, there is leakage.

### 6.6 Acceptance checks
- All leakage tests pass.
- Print the results table for the three baselines.
- Expect a test accuracy of roughly **50–55%**. **If any baseline exceeds 60%, treat it as leakage: find and fix the cause, log it in `PROGRESS.md`, then continue.**

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 7. Phase 4: Market Reaction Signature (daily version first)

**Files:** `src/features/reaction_signature.py`

Write one function that works for both daily and intraday price paths:

```python
def compute_signature(paths: dict[str, pd.Series], event_time, pre_vol: dict[str, float],
                      checkpoints: list[str], expected: dict | None) -> dict[str, float]:
    """paths[asset] = price series from 60 min before the event to the end of the window.
       Returns a flat dict of signature features."""
```

### 7.1 Features (per event)
| Feature | Intraday definition | Daily-version proxy |
|---|---|---|
| `latency_min` | Minutes until \|gold move\| > 2 × pre-event 1-min std (capped at the window length) | Not available → set to NaN, and add the flag `has_intraday=0` |
| `dir_5m, dir_1h, dir_close` | Sign of the gold move at each checkpoint | `dir_close` only (the day-*t* return) |
| `mag_5m, mag_1h, mag_close` | Move ÷ pre-event std | `mag_close` = day-*t* return ÷ 20-day vol |
| `vol_ratio` | Post-event 1-min std ÷ pre-event std | \|day-*t* return\| ÷ 20-day vol |
| `persistence` | move_close ÷ move_1h (clipped to [-2, 2]) | Not available → NaN |
| `xa_<asset>_dir, xa_<asset>_mag` | Same as above at the close, for USD, Equities and Oil | Daily returns ÷ their 20-day vol |
| `gap_gold` | `mag_close − expected_mag_close[event_type]` | Same |
| `dir_mismatch` | 1 if sign(actual) ≠ sign(expected) | Same |
| `event_type` one-hot | FOMC / CPI / NFP | Same |

- **Expected reaction per event type:** the mean of `mag_close` (and of each cross-asset move) over **train-split events only**. Save it to `data/processed/expected_reactions.json`.
- Missing values: impute the train mean and add the indicator columns (`*_missing`).

### 7.2 Signature for non-event days (decay)
For day *t* with the most recent event *e* at day *t_e* ≤ *t*:
`s_t = s_e · exp(−λ · (t − t_e))` with λ = 0.3 from the config, plus the features `days_since_event` and `last_event_type`. If there was no event in the last 10 trading days, `s_t = 0`.

### 7.3 Output
`data/processed/signatures_daily.parquet`: one row per trading day, with the decayed signature columns.

### 7.4 Acceptance checks
- Print the signatures of 3 well-known events, for example the FOMC meetings of 2013-06-19, 2020-03-15 and 2022-06-15. Check the signs make sense: on 2022-06-15 the USD rose.
- Expected reactions are computed from train events only. Assert that no event date used is later than 2019-12-31.
- A unit test on a synthetic path checks that latency, magnitude and persistence come out correctly. Synthetic data is fine here, and labelled as such.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 8. Phase 5: Event text embeddings and the news baseline

**Files:** `src/models/event_encoder.py`, config `baseline_lstm_news`

- Model: `ProsusAI/finbert`, frozen. Embed `text` per event: chunk FOMC statements into ≤ 512 tokens and mean-pool the [CLS] embeddings across chunks. Then reduce 768 → 16 dimensions with **PCA fitted on train events only**.
- Also save FinBERT's sentiment probabilities (positive/negative/neutral) per event as interpretable features.
- Decay the embeddings onto non-event days the same way as the signature (section 7.2).
- Cache the results in `data/processed/event_emb.parquet`. FinBERT runs once; never re-embed on every training run.
- `baseline_lstm_news` = `baseline_lstm` + [decayed event embedding + sentiment] concatenated into the head.
- **Optional, skip by default:** GDELT headlines as an extra text source. Skip it by default.

### Acceptance checks
- The embedding file has one row per event and no NaNs.
- `baseline_lstm_news` trains with 5 seeds, and its results are added to the comparison table.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 9. Phase 6: Static GNN and RC-DGNN (core model)

**Files:** `src/models/dynamic_gnn.py`, `src/models/edge_generator.py`, `src/models/temporal.py`, `src/models/heads.py`, `src/models/rc_dgnn.py`

### 9.1 Architecture (keep exactly this unless the user approves a change)

```
Inputs for day t:
  X_nodes: (B, L=30, N=5, F)      per-node features over the window
  S:       (B, L, D_s)            decayed reaction signature for each day τ in the window
  E:       (B, L, D_e=16+3)       decayed event embedding + sentiment for each τ
  G:       (B, G_dim)             global features for day t (technicals, VIX, regime one-hot)

For each τ in 1..L:
  1. Node encoding:     h_τ = Linear(F → 32)(X_nodes[:, τ])                     # (B, 5, 32)
  2. Edge logits:       A_static (5x5 learnable parameter, shared)
                        ΔA_τ = EdgeGenerator([S_τ, E_τ]) → reshape (B, 5, 5)    # MLP: D → 32 → 25, tanh
                        gate_τ = sigmoid(w · [S_τ, E_τ] + b)                     # how much the event changes the graph
                        logits_τ = A_static + gate_τ * ΔA_τ + attention_scores(h_τ)
                        A_τ = softmax over neighbours (rows), self-loops included
  3. Graph layer x2:    h_τ = ELU(A_τ @ W h_τ) + h_τ   (residual, dropout 0.2)
  4. Read out:          z_τ = concat(h_τ[gold], mean_over_nodes(h_τ))          # (B, 64)

Temporal:  LSTM(64 → 64) over z_1..z_L → last hidden state
Head:      MLP([lstm_out, G] → 32 → 1) → logit;  P(up) = sigmoid(logit)
Uncertainty: MC Dropout, T = 30 forward passes at inference → p_up = mean, p_std = std
Return A_τ for the last day as well (for explainability).
```

- `attention_scores(h)` is a standard GAT-style score: `LeakyReLU(a^T [W h_i || W h_j])`.
- **Static GNN variant:** identical, except `ΔA = 0` and `gate = 0`. Use the config flag `reaction_conditioned: false`.
- **Regularisation:** an L1 penalty on ΔA (weight 1e-3) so the graph changes only when the reaction is informative, plus dropout 0.2–0.3 and weight decay.
- The parameter count must be under 100k. Print it.

### 9.2 Configs
| Config | Graph | Reaction-conditioned | Signature |
|---|---|---|---|
| `static_gnn` | ✓ | ✗ | not used (event embedding still given to the head) |
| `rcdgnn_daily` | ✓ | ✓ | `signatures_daily.parquet` |
| `rcdgnn_full` | ✓ | ✓ | `signatures_intraday.parquet` (Phase 7) |

### 9.3 Acceptance checks
- A forward-pass unit test with random tensors checks the output shape and that the rows of A sum to 1.
- **Overfit test:** the model reaches > 95% training accuracy on 64 samples. This proves it can learn at all.
- `static_gnn` and `rcdgnn_daily` both train with 5 seeds, and their results are added to the table.
- Report the **key comparison**, `static_gnn` vs `rcdgnn_daily`, on all test days **and on post-event days**.
- **Report the result honestly even if RC-DGNN doesn't win.** A null result is a valid finding. Do not tune on the test set to make it win. All tuning happens on validation only.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 10. Phase 7: Intraday reaction signatures

**Files:** `src/data/download_intraday.py`, `src/features/reaction_signature.py` (reuse)

- For each event, download **1-minute (m1)** bars for `xauusd`, a USD proxy (`dollaridxusd` if available, otherwise `eurusd` with its sign inverted), `usa500idxusd` and `brentcmdusd`. The window runs from 60 minutes before the event to 17:00 ET on the event day. Use the instrument IDs verified in Phase 1.
- Cache each window as `data/raw/intraday/<event_id>_<asset>.csv`. The download script must be **resumable**: skip windows that already exist and retry failures 3 times.
- Data quality: if more than 20% of the minutes in a window are missing, flag the event and fall back to that event's daily signature, with `has_intraday=0`.
- Bonds stay daily, because there's no free intraday yield data. Document this as a limitation.
- Output: `data/processed/signatures_intraday.parquet` (same schema as the daily version, with the intraday-only columns filled in).
- Train `rcdgnn_full` with 5 seeds.

### Acceptance checks
- Print the intraday coverage: the % of events with a valid intraday window, per type and year.
- Plot gold's 1-minute path for 3 events with the checkpoints marked: `reports/figures/phase7_event_paths.png`.
- Report the second key comparison: `rcdgnn_daily` vs `rcdgnn_full`.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 11. Phase 8: Evaluation, significance and explainability

**Files:** `src/evaluation/ablation.py`, `src/evaluation/event_windows.py`, `src/evaluation/backtest.py`, `src/explain/shap_explain.py`, `src/explain/graph_viz.py`

### 11.1 Ablation table
`reports/results/ablation_table.csv` and `.md`: every config × {all, event, post-event, calm, volatile, crisis} × {acc, F1, MCC, Brier, log loss, ECE}, as the mean ± std over 5 seeds.

### 11.2 Significance
- **McNemar test** on test-set predictions (seed-ensembled, threshold 0.5) for the pairs `static_gnn` vs `rcdgnn_daily`, `rcdgnn_daily` vs `rcdgnn_full`, and `baseline_lstm` vs `rcdgnn_full`.
- **Diebold–Mariano test** on the Brier-score loss series for the same pairs.
- Report the p-values and say clearly when a result is **not** significant.

### 11.3 Event analysis (`event_windows.py`)
- Metrics restricted to the post-event days, split by event type.
- Stress periods: Feb–Apr 2020 and Mar–Dec 2022.

### 11.4 Calibration
- Reliability diagram per model: `reports/figures/calibration.png`.

### 11.5 Explainability
- **Graph change figure** (the key figure): heatmaps of the 5×5 adjacency `A` the day before vs. the day of 3–4 major events, for example the 2020-03-15 emergency cut, the 2022-06-15 75bp hike and a big CPI surprise. Save to `reports/figures/graph_change_<event_id>.png`.
- **Average edge change by event type:** the mean of (A_event − A_static) for FOMC, CPI and NFP.
- **Feature importance:** permutation importance of the signature features on the val set (main method, cheap and robust). Optionally, SHAP `DeepExplainer` on the head inputs.

### 11.6 Simple backtest (optional, clearly labelled as simplified)
- Long/flat strategy: long gold when `p_up > 0.55`, flat otherwise, with 2 bp of cost per trade.
- Report the Sharpe ratio, max drawdown and hit rate vs. buy-and-hold.
- State in the output that this is **not** evidence of tradable profit.

### 11.7 Acceptance checks
- All figures and tables are generated by one command: `make evaluate`.
- A short `reports/results/SUMMARY.md` written in plain English: what won, by how much, whether it's significant, and where it failed.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 12. Phase 9: Dashboard

**File:** `app/dashboard.py` (Streamlit), run with `streamlit run app/dashboard.py`

Pages and sections:
1. **Forecast:** pick a test date → show P(up) and the uncertainty band, the regime, the last event and days since it, the true outcome, and whether the forecast was right.
2. **Graph:** the 5-node graph drawn with matplotlib or networkx, edge thickness = weight, shown before and after the last event.
3. **Event explorer:** pick an event → the 1-minute gold path, the signature values and the expected vs. actual reaction.
4. **Results:** the ablation table, the calibration plot and the significance results.

The dashboard reads only from saved files in `reports/` and `data/processed/`, and never trains anything. It must load in under 5 seconds.

**CHECKPOINT: append this phase's report to `PROGRESS.md`, commit, then continue straight to the next phase.**

---

## 13. Phase 10: Documentation and cleanup

- Update `README.md` with the final results table, key figures, exact data sources and dates, and the limitations, including the consensus-forecast and bond-intraday gaps.
- Add docstrings to every public function. `pytest` passes, and `make all` reproduces everything from the raw data.
- Write `docs/VIVA_NOTES.md`: a one-page explanation of each module and the 10 most likely viva questions, answered using the **actual results**.

---

## 14. Final repo layout (target)

```
gold-reaction/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── Makefile
├── requirements.txt / requirements-lock.txt
├── .env (not committed)
├── configs/        base.yaml + baseline_majority, baseline_logreg, baseline_lstm,
│                   baseline_lstm_news, static_gnn, rcdgnn_daily, rcdgnn_full
├── data/raw/{daily,intraday,events}/  data/interim/  data/processed/
├── src/
│   ├── utils/      config.py, seed.py, paths.py
│   ├── data/       download_daily.py, download_intraday.py, download_events.py, align.py
│   ├── features/   technical.py, regime.py, reaction_signature.py
│   ├── models/     event_encoder.py, edge_generator.py, dynamic_gnn.py, temporal.py,
│   │               heads.py, rc_dgnn.py, baselines.py
│   ├── training/   dataset.py, splits.py, train.py
│   ├── evaluation/ metrics.py, ablation.py, event_windows.py, backtest.py
│   └── explain/    shap_explain.py, graph_viz.py
├── app/dashboard.py
├── tests/          test_leakage.py, test_features.py, test_models.py
├── reports/figures/  reports/results/
└── docs/VIVA_NOTES.md
```

---

## 15. Known risks and what to do

| Risk | Action |
|---|---|
| Dukascopy instrument missing or download blocked | Report it. For gold, fall back to `GC=F` with the 13:30 ET cutoff. For the others, use the daily proxy and `has_intraday=0`. |
| federalreserve.gov page layout differs by year | Use separate parsers per era, and save the raw HTML. If parsing fails, try another page or the press-release archive; log any event you could not get. Never hand-enter dates from memory. |
| FRED vintage dates include revision dates | Filter to about 1 release per month, and inspect years with odd counts. |
| A model gets > 60% test accuracy | Assume leakage until proven otherwise. Re-run the leakage tests and the shuffled-label test. |
| RC-DGNN does not beat the static GNN | Report it honestly. Analyse the post-event days and the edge changes, which are still a valid contribution. |
| Validation loss diverges / overfitting | Reduce the hidden size to 16, increase dropout, and strengthen the L1 penalty on ΔA. Report what changed. |
| Class imbalance in the 2024–26 test period | Always report MCC and Brier score next to accuracy, and include the majority baseline. |

---

## 16. Suggested team split (for the humans)

| Person | Phases owned | Reviews |
|---|---|---|
| Hitha | 1, 3 (data, baselines, leakage tests) | 6 |
| Hrithik | 2, 5 (events, FinBERT, regimes) | 7 |
| Mahek | 4, 7 (reaction signature, intraday) | 3 |
| All | 6 (RC-DGNN), 8 (evaluation), 9–10 | — |