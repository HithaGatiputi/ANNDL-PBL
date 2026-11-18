# RC-DGNN: Reaction-Conditioned Dynamic GNN for Event-Aware Gold Price Forecasting

ANN Project-Based Learning · RV College of Engineering, Dept. of CSE (AI & ML)

| USN | Name |
|---|---|
| 1RV24CI040 | Hitha Gatiputi |
| 1RV24CI041 | Hrithik Bhattacharya |
| 1RV24CI062 | Mahek Arora |

> A gold forecaster that measures **how the market actually reacts in the first hours after an event**, turns that reaction into a **Market Reaction Signature**, and uses it to **dynamically update the relationships between Gold, USD, Bonds, Equities and Oil** before predicting gold's next-day direction.

---

## 1. Problem

Gold moves sharply on events: Fed decisions, inflation data, jobs reports, wars, crises. Most forecasting models:
- use **price history only** and see the event only after the price has moved, or
- add **news sentiment as a static input**: *what the news said*, not *how the market responded*, or
- use **GNNs with fixed or slowly-learned asset relationships** that do not change in response to a specific event.

**Research question:** Does the early market reaction to an event, used to reweight cross-asset relationships, improve next-day gold direction forecasts compared to price-only and static-graph models?

## 2. Research Gaps Targeted

| # | Gap | Our answer |
|---|-----|-----------|
| 1 | **Static event integration:** news is fused by its content, not by how the market reacted to it. | Event embedding is combined with the **observed reaction**, not used alone. |
| 2 | **Reaction info is descriptive:** event studies measure gold's reaction timing, magnitude and volatility statistically but don't use it as a learned input. | Reaction trajectory is converted into a **Market Reaction Signature** used as a learned state. |
| 3 | **Relationships not reaction-conditioned:** dynamic/event-driven GNNs exist, but our targeted survey found no gold-specific model that reweights the asset graph by the observed event reaction. | Signature generates the **edge weights** of a dynamic asset graph. |

**Novelty = Reaction-Conditioned Dynamic GNN for gold.** **Result: on 2022–2026 test data the reaction conditioning did not improve forecasts (null result, §9).** We do not claim news sentiment or uncertainty alone is new (CNN-QRLSTM, Ji et al., 2026, already does both for gold).

## 3. Key Concept: Market Reaction Signature

For each event (e.g. FOMC at 2:00 PM New York), we measure the early market response using **1-minute data** from 1 hour before the event to that day's close:

| Feature | Definition |
|---|---|
| **Latency** | Minutes until gold's move exceeds 2× its pre-event volatility |
| **Direction** | Sign of gold move at +5 min, +1 h, close |
| **Magnitude** | Size of move at +5 min, +1 h, close (in pre-event std units) |
| **Volatility change** | Post-event volatility ÷ pre-event volatility |
| **Persistence / decay** | Fraction of the +1 h move still present at the close |
| **Cross-asset response** | Same measures for USD, Equities, Oil (direction + magnitude) |
| **Reaction gap** | Actual gold move − *expected* move for that event type (historical average, **training data only**) |

```
Event: "Fed raises rates"
Expected:  USD ↑  Yields ↑  Gold ↓
Actual:    USD ↑  Yields ↑  Gold ↑ (fast, persistent)
                       ↓
     Signature → strengthens Gold–safe-haven links, weakens Gold–USD link
```

Between events, the signature decays: `s_t = s_event · exp(−λ · days_since_event)`, plus a `days_since_event` feature.

**Why intraday:** daily bars mix the event with everything else that happened that day. Markets absorb macro news within minutes (Andersen et al., 2003), and patterns like *spike-then-reversal* vs. *slow drift* are only visible intraday. Everything else stays daily.

## 4. Architecture (as built)

```
For each day τ of a 30-day window (B = batch, N = 5 nodes: Gold, USD, Bonds, Equities, Oil):
  h_τ       = Linear(F=3 → 16)(X_nodes[:, τ])                        node features: return/Δyield, 5-day return, 20-day vol
  ΔA_τ,gate = EdgeGenerator([S_τ, E_τ])                              S = decayed reaction signature, E = FinBERT emb + sentiment
  A_τ       = softmax_rows(A_static + gate·ΔA_τ + GAT(h_τ))          5×5, self-loops included
  h_τ       = 2 × [ELU(A_τ W h_τ) + h_τ]                             residual graph layers (plain PyTorch, no torch-geometric)
  z_τ       = [h_τ(gold), mean(h_τ)]
LSTM(32 → 16) over z_1..z_30 → MLP([lstm, G, E_t] → 16 → 1) → P(up);  MC dropout (T = 30) → p_std
Static GNN = identical with ΔA = 0, gate = 0.   Loss: BCE + 0.01·|gate·ΔA|.   6,083 parameters.
```

| Module | File |
|---|---|
| Config / seed / paths | `src/utils/{config,seed,paths}.py` |
| Downloads | `src/data/download_daily.py`, `download_intraday.py`, `download_events.py`, `align.py` |
| Features | `src/features/technical.py`, `regime.py`, `reaction_signature.py`, `intraday_signature.py` |
| Models | `src/models/event_encoder.py`, `edge_generator.py`, `dynamic_gnn.py`, `temporal.py`, `heads.py`, `rc_dgnn.py`, `baselines.py` |
| Training | `src/training/{dataset,splits,train}.py` |
| Evaluation | `src/evaluation/{metrics,ablation,event_windows,backtest,summary,common}.py` |
| Explainability | `src/explain/graph_viz.py`, `shap_explain.py` (permutation importance) |
| Dashboard | `app/dashboard.py` |

## 5. Data (exact sources)

| Data | Resolution | Source |
|---|---|---|
| Gold XAU/USD (target) | h1 → **17:00 New York close** | Dukascopy via `npx dukascopy-node` (`xauusd`, bid) |
| Gold, USD, Equities, Oil around events | 1-minute, event − 60 min → 17:00 ET | Dukascopy `xauusd`, `dollaridxusd` (EUR/USD `eurusd` inverted where the index has no data, mostly 2013–17), `usa500idxusd`, `brentcmdusd` |
| USD index, S&P 500, Brent, VIX | daily | Yahoo Finance: `DX-Y.NYB`, `^GSPC`, `BZ=F`, `^VIX` |
| US 10Y yield | daily (Δ in bp) | FRED `DGS10` |
| FOMC statements + release times | event | federalreserve.gov press releases `monetaryYYYYMMDDa.htm` (111 statements incl. unscheduled 2019-10-11, 2020-03-03, 2020-03-15, 2020-03-23) |
| CPI / NFP release dates, first-release values | event, 08:30 ET | ALFRED/FRED vintages of `CPIAUCSL` / `PAYEMS` (161 each) |

**Period:** 2013-01-02 → 2026-06-30, **3,481 trading days**, **433 events**. Splits: train 2013–2019 (1,805 days), val 2020–2021 (517), test 2022-01 → 2026-06 (1,159). Raw data are not committed; `make data` downloads them (resumable, cached).

## 6. Leakage rules (enforced by `tests/test_leakage.py`)

1. Every feature for day t uses data timestamped ≤ 17:00 ET on day t; the target is close t → close t+1 (tested by deleting future rows and asserting identical features).
2. Scalers, regime thresholds, the HMM, expected reactions, signature scaling and the FinBERT PCA are fitted on train only (tested by perturbing val/test data).
3. HMM regimes use forward filtering only (no Viterbi / smoothing).
4. Intraday windows end at 17:00 ET of the event's trading day; events at/after 17:00 or on non-trading days move to the next trading day.
5. Shuffled-label test: an LSTM trained on permuted labels scores 0.517/0.527 vs the 0.533 majority rate.

## 7. Regime detection

20-day gold volatility vs train percentiles (calm < 33 < normal < 67 < volatile, > 95 crisis) plus a 3-state Gaussian HMM check. Agreement Cohen's κ = 0.60; March 2020 is 100% volatile/crisis under both.

## 8. Experiments

| Config | Price | Event emb | Graph | Reaction-conditioned | Signature |
|---|:-:|:-:|:-:|:-:|:-:|
| `baseline_majority` | | | | | |
| `baseline_logreg` | ✓ (gold lags + globals) | | | | |
| `baseline_lstm` | ✓ | | | | |
| `baseline_lstm_news` | ✓ | ✓ | | | |
| `static_gnn` | ✓ | ✓ (head) | static | | |
| `rcdgnn_daily` | ✓ | ✓ | dynamic | ✓ | daily proxy |
| `rcdgnn_full` | ✓ | ✓ | dynamic | ✓ | **1-minute intraday** (98.4% of events) |

5 seeds per neural model; early stopping on validation log loss; all tuning on validation only.

## 9. Results (test set 2022-01-03 → 2026-06-30, mean ± std over 5 seeds)

| model | acc (all) | MCC (all) | Brier (all) | ECE (all) | acc (post-event, n=141) | MCC (post-event) |
|---|---|---|---|---|---|---|
| `baseline_majority` | 0.533 ± 0.000 | 0.000 ± 0.000 | 0.250 ± 0.000 | 0.027 ± 0.000 | 0.461 ± 0.000 | 0.000 ± 0.000 |
| `baseline_logreg` | 0.517 ± 0.000 | 0.033 ± 0.000 | 0.249 ± 0.000 | 0.035 ± 0.000 | 0.504 ± 0.000 | 0.005 ± 0.000 |
| `baseline_lstm` | 0.538 ± 0.005 | 0.036 ± 0.024 | 0.249 ± 0.000 | 0.026 ± 0.007 | 0.489 ± 0.009 | 0.076 ± 0.052 |
| `baseline_lstm_news` | 0.535 ± 0.011 | 0.044 ± 0.026 | 0.249 ± 0.001 | 0.026 ± 0.005 | 0.495 ± 0.020 | 0.034 ± 0.042 |
| `static_gnn` | 0.525 ± 0.015 | 0.014 ± 0.047 | 0.249 ± 0.001 | 0.020 ± 0.011 | 0.492 ± 0.057 | 0.031 ± 0.113 |
| `rcdgnn_daily` | 0.517 ± 0.010 | 0.017 ± 0.009 | 0.250 ± 0.000 | 0.026 ± 0.010 | 0.492 ± 0.029 | 0.013 ± 0.053 |
| `rcdgnn_full` | 0.516 ± 0.010 | 0.016 ± 0.009 | 0.250 ± 0.000 | 0.026 ± 0.010 | 0.489 ± 0.030 | 0.005 ± 0.058 |

**Findings**
- **Null result.** Reaction conditioning does not beat the static graph: static vs RC-DGNN-daily McNemar p = 1.00, Diebold–Mariano p = 0.39; on post-event days p = 0.26 / 0.89. Intraday vs daily signature: p = 1.00 / 0.30. No pairwise difference is significant at 5% (`reports/results/significance.md`).
- No model clearly beats "always up" (0.533); MCCs ≈ 0. Next-day gold direction is close to unpredictable from these inputs.
- The learned graph barely responds to events (largest mean event-driven edge change −0.008 on weights ≈ 0.2). Removing the L1 penalty makes ΔA ~5× larger without changing validation loss, so the null result is not caused by the penalty.
- Simplified long/flat backtest: no model beats buy-and-hold over the 2022–26 gold rally (not evidence of tradable profit).
- Full write-up: `reports/results/SUMMARY.md`; build log and every decision: `PROGRESS.md`.

**Key figures** (`reports/figures/`): `graph_change_FOMC_20220615.png`, `graph_change_FOMC_20200315.png`, `graph_change_by_type.png`, `phase7_event_paths.png`, `calibration.png`, `permutation_importance.png`, `phase2_regimes.png`, `phase1_prices.png`.

## 10. Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Node.js LTS also needed (npx dukascopy-node)
echo "FRED_API_KEY=your_key" > .env        # optional: keyless FRED/ALFRED endpoints are used otherwise
make all                                   # data → features → 7 configs × 5 seeds → evaluate → tests
make dashboard                             # streamlit run app/dashboard.py
```

Other targets: `make data`, `make features`, `make train VARIANT=rcdgnn_full`, `make ablation`, `make evaluate`, `make test`.

## 11. Repo structure

```
configs/   base.yaml + one YAML per variant (configs/tuning/ = validation-only tuning/diagnostic runs)
data/      raw/{daily,intraday,events}  interim/  processed/      (git-ignored, rebuilt by make data/features)
src/       utils/ data/ features/ models/ training/ evaluation/ explain/
app/       dashboard.py
tests/     test_leakage.py, test_features.py, test_models.py
reports/   figures/  results/ (per-variant seed predictions, tables, SUMMARY.md)
docs/      VIVA_NOTES.md
```

## 12. Limitations

- Daily gold direction is very noisy; differences between models are within seed noise and not significant.
- **No consensus forecasts**: the CPI/NFP "surprise" is actual vs the prior month, not vs economist consensus (not freely available). The CPI/NFP text fed to FinBERT is a template, so its embedding carries little meaning.
- **Bonds have no intraday data** (no free intraday yields); bonds stay daily. Dukascopy's dollar index has gaps before 2018, so 172 events use inverted EUR/USD as the USD proxy.
- 433 events give little training signal to the edge generator; the planned 32/64 sizes overfitted and were reduced to 16 (validation-chosen).
- Latency as defined (|move| > 2 × pre-event 1-min σ) is ~1 minute for most events, so it is barely informative.
- FOMC times for 2013–2015 are not printed on the press releases; 14:00 ET was verified from hourly gold volatility.
- Regime labels are self-derived; the backtest ignores spreads, slippage and financing. Decision-support prototype, **not** a trading system.

## 13. References

- Amini & Kalantari (2024). Gold price prediction by a CNN-Bi-LSTM model with automatic parameter tuning. *PLOS ONE*. doi:10.1371/journal.pone.0298426
- Baradaran, Bohlouli & Jahed Motlagh (2023). Decoding tomorrow's gold prices: GRU vs CNN-LSTM in the Iranian market.
- Varela (2024). Achilles: Neural network to predict gold vs USD with trading bot. arXiv:2410.21291
- Ji, Lei, Zhang, Heng & Fan (2026). Forecasting the price of gold with integrated media sentiment: CNN-QRLSTM. *Entropy* 28(3), 271. doi:10.3390/e28030271
- Gong (2024). Gold price forecasting based on LSTM and linear regression. *SHS Web of Conferences*.
- Andersen, Bollerslev, Diebold & Vega (2003). Micro effects of macro announcements: real-time price discovery in foreign exchange. *American Economic Review* 93(1).
- Araci (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063
- *TODO: add papers backing Gaps 1–3 (event fusion, gold event studies, dynamic/event-driven GNNs).*

