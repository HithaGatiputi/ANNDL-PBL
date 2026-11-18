# VIVA NOTES — RC-DGNN for event-aware gold forecasting

## Modules in one line each
- **`src/data/download_daily.py`** — Yahoo (DXY, S&P 500, Brent, VIX) and FRED DGS10, saved raw and never modified.
- **`src/data/download_intraday.py`** — calls `npx dukascopy-node`: gold hourly bars (for the 17:00 ET close) and 1-minute windows for every event (resumable, 3 retries; EUR/USD fallback for USD).
- **`src/data/align.py`** — gold 17:00 ET close from hourly bars; other series forward-filled ≤ 2 days onto gold trading days; target `y_next`.
- **`src/data/download_events.py`** — FOMC statements and real release times scraped from federalreserve.gov; CPI/NFP release dates and first-release values from ALFRED vintages; maps each event to its trading day.
- **`src/features/regime.py`** — volatility-percentile regimes (train thresholds) and a forward-filtered 3-state HMM check.
- **`src/features/reaction_signature.py`** — `compute_signature`: direction/magnitude at +5m/+1h/close, latency, volatility ratio, persistence, cross-asset moves, gap vs the train-mean expected reaction; decays it onto later days (λ = 0.3).
- **`src/features/intraday_signature.py`** — loads 1-minute windows, checks quality (> 20% missing → daily fallback).
- **`src/models/event_encoder.py`** — frozen FinBERT: chunked [CLS] mean-pool, sentiment, PCA 768→16 on train events.
- **`src/models/{edge_generator,dynamic_gnn,temporal,heads,rc_dgnn}.py`** — the model: ΔA from [signature, embedding] with a gate, GAT-style attention, 2 residual graph layers, LSTM, MLP head, MC dropout.
- **`src/models/baselines.py`** — majority, logistic regression, LSTM (± event embedding).
- **`src/training/{dataset,splits,train}.py`** — 30-day windows, train-only scaling, fixed time splits, AdamW + early stopping on val log loss, 5 seeds.
- **`src/evaluation/*`** — metrics (acc, F1, MCC, Brier, log loss, ECE) per subset, ablation table, McNemar + Diebold–Mariano, event/stress analysis, calibration, backtest, SUMMARY.
- **`src/explain/*`** — adjacency heatmaps before/after events, event-driven ΔA by type, permutation importance.
- **`tests/`** — leakage (10 tests), features (synthetic paths), models (shapes, rows sum to 1, overfit 64 samples).

## 10 likely questions (answered with our actual results)

1. **Did RC-DGNN beat the static GNN?** No. Test accuracy 0.517 ± 0.010 (daily signature) and 0.516 ± 0.010 (intraday) vs 0.525 ± 0.015 for the static GNN. McNemar p = 1.00, DM p = 0.39. On post-event days (n = 141, seed-ensembled): RC-DGNN-daily 0.511 vs static 0.461, McNemar p = 0.26 — not significant. It's a null result, and we report it as one.

2. **Is anything better than predicting "always up"?** Not meaningfully. The majority baseline scores 0.533. The best model, the LSTM, scores 0.538 ± 0.005 with MCC 0.036, and LSTM vs RC-DGNN-full has p = 0.86. Brier scores are about 0.249–0.250, the same as a coin flip.

3. **How do you know there is no leakage?** `tests/test_leakage.py` checks four things. Deleting all rows after day t leaves day-t features identical. Perturbing val/test data leaves the scalers, thresholds, HMM, expected reactions and PCA unchanged. Split target periods don't overlap. Training on shuffled labels gives 0.517/0.527, below the 0.533 majority rate. No model exceeded 60%.

4. **Why is the target close at 17:00 New York, and why does the reaction window end there?** That is the end of the gold trading day. The reaction window (event − 60 min → 17:00 ET on day t) ends exactly where the target (close t → close t+1) starts, so there is no overlap.

5. **What does the graph learn on event days?** Very little. The largest average event-driven edge change on test event days is −0.008, against weights of about 0.2. The day-to-day changes in A come mostly from the attention scores, which follow that day's prices. Removing the L1 penalty makes ΔA about 5× larger but leaves validation loss the same (0.6927), so the events simply don't carry usable next-day signal here.

6. **Why did you shrink the model?** At the planned 32/64 hidden sizes (~40k parameters), validation loss rose from epoch 0–1 while training loss fell, which is overfitting. Following the plan's fallback, we cut the hidden size to 16, raised dropout to 0.4 and the L1 penalty to 0.01. We chose this on mean validation log loss (0.6929 → 0.6924) and never on test.

7. **Did intraday data help over the daily proxy?** No. 98.4% of events have a valid 1-minute gold window, but rcdgnn_full vs rcdgnn_daily differ by under 0.001 in P(up). Accuracy 0.516 vs 0.517, McNemar p = 1.00.

8. **What does the signature show for famous events?** Taper tantrum, 2013-06-19: gold −1.11σ, USD +2.09σ, equities −1.61σ. FOMC 2022-06-15 (75 bp hike): gold dipped in the first 5 minutes, then rose by +1 h and the close, and the USD fell by 17:00. The plan expected the USD to rise; the data show it fell.

9. **How are "expected reactions" and the surprise defined?** The expected reaction is the mean close magnitude per event type over the 225 train events (2013–2019): FOMC +0.21σ, CPI +0.20σ, NFP +0.11σ. The gap is actual minus expected. For CPI/NFP the "surprise" is actual vs the prior month, because consensus forecasts aren't freely available — a stated limitation.

10. **What are the main limitations, and what would you do next?** Daily direction is noisy, and 141 post-event test days are too few to detect small effects. There are no consensus forecasts, no intraday bond yields, and the CPI/NFP text is a template. Next steps: predict volatility or intraday/next-session returns (where reactions matter more), use consensus surprises, and run event studies at shorter horizons.
