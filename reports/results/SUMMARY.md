# SUMMARY (plain English)

Task: predict whether gold (XAU/USD, 17:00 New York close) closes higher on the next trading day. Test period 2022-01-03 → 2026-06-30 (1,159 days). All numbers are mean ± std over 5 seeds unless stated; tuning used the validation split only.

## What won

- Highest test accuracy: **baseline_lstm** (0.538 ± 0.005). Always predicting 'up' (majority baseline) scores 0.533.
- Best probability quality (lowest Brier): **baseline_logreg** (0.249 ± 0.000); a constant 50/50 forecast has Brier 0.250.

| model | acc (all) | MCC (all) | Brier (all) | acc (post-event) | MCC (post-event) |
|---|---|---|---|---|---|
| baseline_majority | 0.533 ± 0.000 | 0.000 ± 0.000 | 0.250 ± 0.000 | 0.461 ± 0.000 | 0.000 ± 0.000 |
| baseline_logreg | 0.517 ± 0.000 | 0.033 ± 0.000 | 0.249 ± 0.000 | 0.504 ± 0.000 | 0.005 ± 0.000 |
| baseline_lstm | 0.538 ± 0.005 | 0.036 ± 0.024 | 0.249 ± 0.000 | 0.489 ± 0.009 | 0.076 ± 0.052 |
| baseline_lstm_news | 0.535 ± 0.011 | 0.044 ± 0.026 | 0.249 ± 0.001 | 0.495 ± 0.020 | 0.034 ± 0.042 |
| static_gnn | 0.525 ± 0.015 | 0.014 ± 0.047 | 0.249 ± 0.001 | 0.492 ± 0.057 | 0.031 ± 0.113 |
| rcdgnn_daily | 0.517 ± 0.010 | 0.017 ± 0.009 | 0.250 ± 0.000 | 0.492 ± 0.029 | 0.013 ± 0.053 |
| rcdgnn_full | 0.516 ± 0.010 | 0.016 ± 0.009 | 0.250 ± 0.000 | 0.489 ± 0.030 | 0.005 ± 0.058 |

## Main hypothesis: does reaction conditioning help?

- static_gnn vs rcdgnn_daily (all, n=1159, seed-ensembled): accuracy 0.528 vs 0.528 (tie), McNemar p = 1.000; Brier 0.2490 vs 0.2494, Diebold–Mariano p = 0.386 → **not significant**.
- static_gnn vs rcdgnn_daily (post_event, n=141, seed-ensembled): accuracy 0.461 vs 0.511 (rcdgnn_daily higher), McNemar p = 0.265; Brier 0.2517 vs 0.2520, Diebold–Mariano p = 0.895 → **not significant**.
- rcdgnn_daily vs rcdgnn_full (all, n=1159, seed-ensembled): accuracy 0.528 vs 0.529 (rcdgnn_full higher), McNemar p = 1.000; Brier 0.2494 vs 0.2495, Diebold–Mariano p = 0.305 → **not significant**.
- rcdgnn_daily vs rcdgnn_full (post_event, n=141, seed-ensembled): accuracy 0.511 vs 0.504 (rcdgnn_daily higher), McNemar p = 1.000; Brier 0.2520 vs 0.2519, Diebold–Mariano p = 0.319 → **not significant**.
- baseline_lstm vs rcdgnn_full (all, n=1159, seed-ensembled): accuracy 0.532 vs 0.529 (baseline_lstm higher), McNemar p = 0.859; Brier 0.2490 vs 0.2495, Diebold–Mariano p = 0.468 → **not significant**.
- baseline_lstm vs rcdgnn_full (post_event, n=141, seed-ensembled): accuracy 0.454 vs 0.504 (rcdgnn_full higher), McNemar p = 0.265; Brier 0.2513 vs 0.2519, Diebold–Mariano p = 0.802 → **not significant**.

Reaction-conditioned (intraday) minus static GNN accuracy: -0.008 on all test days, -0.003 on post-event days. None of the pairwise differences is statistically significant at 5%: **the main hypothesis is not supported by this data**. This is a valid null result.

## What the graph learned

- The largest average event-driven edge change on test event days is -0.0077 (FOMC: Equities → Bonds), on attention weights of about 0.2. The learned graph barely reacts to events: the L1 penalty and the weak signal keep ΔA close to zero.

- Diagnostic (not used for model choice): removing the L1 penalty makes the event-driven edge change ~5x larger but leaves validation loss unchanged (rcdgnn_daily_l1zero: val log loss 0.6927, test acc 0.515; rcdgnn_full_l1zero: val log loss 0.6928, test acc 0.515). So the null result is not caused by the penalty.

- Most important event inputs on validation (permutation, Δ log loss): event_embedding(16) (+0.00030), finbert_sentiment(3) (+0.00003), vol_ratio (+0.00000).

## Where it failed

- No model clearly beats the majority-class rate; MCCs are close to 0, so the daily direction of gold is close to unpredictable from these inputs.
- The 2022–2026 gold rally makes 'always up' hard to beat; post-event days (n=141) are too few to detect small effects.
- See `event_analysis.md` (by event type and stress period), `backtest.md` (simplified, not evidence of tradable profit) and `calibration.png`.

Intraday coverage: 98% of events had a valid 1-minute gold window.
