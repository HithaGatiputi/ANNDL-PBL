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

**Novelty = Reaction-Conditioned Dynamic GNN for gold.** We do not claim news sentiment or uncertainty alone is new (CNN-QRLSTM, Ji et al., 2026, already does both for gold).

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

## 4. Architecture

```
1. Inputs
   ├─ Events/news (FOMC, CPI, NFP text + GDELT) ──► 2. FinBERT (frozen) ──► event embedding
   ├─ Intraday windows around events ───────────► 3. Reaction analysis
   ├─ Daily gold OHLCV + technical indicators       │
   ├─ Daily cross-asset: USD, Bonds, Equities, Oil  ▼
   └─ Macro indicators                          4. Reaction Signature (+ event embedding)
                                                    │
                                                    ▼  edge-weight generator (MLP)
   Daily node features (30-day window) ───────► 5. Dynamic asset graph
        [Gold, USD, Bonds, Equities, Oil]           A_t = A_static + ΔA(signature)
                                                    │
                                                    ▼
                                                6. Graph layers (GAT/GCN) per day
                                                    → Temporal module (LSTM / Transformer)
                                                    │
   Regime detector (calm/normal/volatile/crisis) ──┤
                                                    ▼
                                                7. Output
                                                   • P(gold up next day)
                                                   • Uncertainty (MC Dropout)
                                                   • Explanation (graph attention + SHAP)
```

| Module | File | Notes |
|---|---|---|
| Event encoder | `src/models/event_encoder.py` | FinBERT (`ProsusAI/finbert`), frozen; precompute embeddings |
| Reaction signature | `src/features/reaction_signature.py` | Computed from intraday windows; daily fallback |
| Edge generator | `src/models/edge_generator.py` | MLP: signature + event emb → ΔA (5×5) |
| Dynamic GNN | `src/models/dynamic_gnn.py` | 5 nodes; GAT/GCN using A_t |
| Temporal module | `src/models/temporal.py` | LSTM (main) or small Transformer |
| Regime detector | `src/features/regime.py` | Volatility percentiles + HMM check; input feature |
| Heads | `src/models/heads.py` | Direction probability; MC Dropout uncertainty |
| Full model | `src/models/rc_dgnn.py` | Wires everything together |
| Baselines | `src/models/baselines.py` | LSTM, LSTM+FinBERT, static GNN |

## 5. Data

| Data | Resolution | Source | Notes |
|---|---|---|---|
| Gold (XAU/USD) | **Intraday** (event windows) | Dukascopy (`dukascopy-python`) or HistData.com | 1-minute bars |
| USD (EUR/USD or DXY proxy), Equities (US500), Oil | **Intraday** (event windows) | Dukascopy / HistData.com | Verify instrument availability |
| Gold, DXY, S&P 500, Oil | Daily | Yahoo Finance (`yfinance`) | `GC=F`, `DX-Y.NYB`, `^GSPC`, `CL=F` |
| Bonds (US 10Y yield) | Daily | FRED (`DGS10`) | No free intraday yields: limitation |
| Macro indicators | Monthly/daily | FRED | CPI, Fed funds rate, etc. |
| FOMC dates + statements | Event | federalreserve.gov | 2:00 PM New York; text for FinBERT |
| CPI / NFP release dates | Event | BLS release calendar | 8:30 AM New York |
| Geopolitical / general news | Event | GDELT 2.0 (2015+) | 15-min timestamps, tone scores |

**Period:** 2013-01-01 to 2026-06-30 (~3,400 trading days). **Events:** ~30/year (8 FOMC + 12 CPI + 12 NFP) → ~400 events, plus GDELT spikes.

Intraday data is downloaded **only for event windows**, not the full history.

## 6. Leakage Rules (non-negotiable)

1. **Reaction window ends at the day-t close; target is the day t → t+1 direction.** No overlap.
2. **Time-ordered splits only:** Train 2013–2019, Val 2020–2021, Test 2022-01 to 2026-06.
3. **News cutoff:** published before the day-t close (17:00 New York) → day t; after → day t+1. Date-only sources → lag 1 day.
4. **Fitted on training data only:** expected reactions per event type, scalers, regime thresholds, HMM, static adjacency.
5. `tests/test_leakage.py` asserts every feature timestamp < target start.

## 7. Regime Detection (supporting module)

- **Main:** 20-day rolling volatility → train-set percentiles: calm / normal / volatile, top 5% crisis.
- **Check:** `hmmlearn.GaussianHMM` (3 states) on returns + volatility.
- **Validation:** both should flag March 2020 and 2022 as volatile/crisis.
- Used as an input feature and for per-regime evaluation.

## 8. Experiments (Ablation)

| Config | Price | Event emb | Graph | Reaction-conditioned | Signature data |
|---|:-:|:-:|:-:|:-:|:-:|
| `baseline_lstm` | ✓ | | | | |
| `baseline_lstm_news` | ✓ | ✓ | | | |
| `static_gnn` | ✓ | ✓ | static | | |
| `rcdgnn_daily` | ✓ | ✓ | dynamic | ✓ | daily |
| `rcdgnn_full` | ✓ | ✓ | dynamic | ✓ | **intraday** |

Key comparisons:
- `static_gnn` vs `rcdgnn_*` → **does reaction conditioning help?** (the main novelty test)
- `rcdgnn_daily` vs `rcdgnn_full` → **does intraday reaction data help?**

Each config run with 5 seeds; report mean ± std.

## 9. Evaluation

- **Direction:** accuracy, F1, MCC
- **Probability quality:** Brier score, log loss, calibration curve / ECE
- **Event days vs. non-event days:** main gains expected on days after events
- **Stress periods:** COVID (Mar 2020), rate hikes (2022)
- **By regime:** calm / volatile / crisis
- **Significance:** McNemar or Diebold–Mariano test vs. baselines
- **Optional:** simple long/flat backtest (Sharpe, max drawdown), costs simplified

Realistic expectation: **52–56% direction accuracy overall**, with the main claim being better behaviour on post-event days.

## 10. Explainability

- **Graph attention / edge weights:** how the Gold–USD, Gold–Bonds, etc. links changed after each event (key figure for the report)
- **SHAP** on signature features: which part of the reaction drove the call
- **Dashboard** (`app/dashboard.py`, Streamlit): prediction, confidence, regime, graph before/after event, top drivers

## 11. Repo Structure

```
gold-reaction-nn/
├── configs/            # base + one YAML per ablation variant
├── data/
│   ├── raw/            # daily/, intraday/, events/, news/  (git-ignored)
│   ├── interim/        # cleaned, aligned
│   └── processed/      # model-ready tensors, FinBERT embeddings, signatures
├── notebooks/          # 01 exploration → 06 evaluation
├── src/
│   ├── data/           # download_daily, download_intraday, download_events, download_news, align
│   ├── features/       # technical, regime, reaction_signature
│   ├── models/         # event_encoder, edge_generator, dynamic_gnn, temporal, heads, rc_dgnn, baselines
│   ├── training/       # dataset, splits, train
│   ├── evaluation/     # metrics, ablation, event_windows, backtest
│   └── explain/        # shap_explain, graph_viz
├── app/                # Streamlit dashboard
├── scripts/            # run_pipeline.sh
├── tests/              # leakage + feature tests
└── reports/            # figures/, results/
```

## 12. Build Order

| Phase | Task | Done when |
|---|---|---|
| 1 | Daily data: download + align (Yahoo, FRED) | Clean daily table, no gaps |
| 2 | Event calendar (FOMC/CPI/NFP) + regime labels | Event table with exact times; regimes validated |
| 3 | **Baseline LSTM** | Test metrics logged |
| 4 | Reaction signature from **daily** data | `rcdgnn_daily` inputs ready |
| 5 | FinBERT event embeddings + `baseline_lstm_news` | Runs |
| 6 | **Static GNN** → **RC-DGNN (daily signature)** | Both run; main novelty test done |
| 7 | Intraday data (Dukascopy) → intraday signature → `rcdgnn_full` | Runs |
| 8 | Ablation, event-day analysis, significance, explainability | Result tables + figures |
| 9 | Dashboard + report | Demo ready |

**Safety net:** phases 1–6 give a complete RC-DGNN project on daily data. Phase 7 upgrades it with intraday reactions.

## 13. Setup

```bash
git clone <repo-url> && cd gold-reaction-nn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "FRED_API_KEY=your_key" > .env
bash scripts/run_pipeline.sh
streamlit run app/dashboard.py
```

## 14. Limitations

- Daily gold direction is very noisy; gains over baselines may be small.
- ~400 events → limited training signal for the edge generator; keep it small and regularised.
- No free intraday bond yields; bonds use daily FRED data.
- USD intraday uses EUR/USD or a DXY proxy depending on availability.
- "Expected reaction" depends on how event types are defined.
- Regime labels are self-derived, not ground truth.
- Backtest simplifies costs and slippage. Decision-support prototype, **not** a trading system.

## 15. References

- Amini & Kalantari (2024). Gold price prediction by a CNN-Bi-LSTM model with automatic parameter tuning. *PLOS ONE*. doi:10.1371/journal.pone.0298426
- Baradaran, Bohlouli & Jahed Motlagh (2023). Decoding tomorrow's gold prices: GRU vs CNN-LSTM in the Iranian market.
- Varela (2024). Achilles: Neural network to predict gold vs USD with trading bot. arXiv:2410.21291
- Ji, Lei, Zhang, Heng & Fan (2026). Forecasting the price of gold with integrated media sentiment: CNN-QRLSTM. *Entropy* 28(3), 271. doi:10.3390/e28030271
- Gong (2024). Gold price forecasting based on LSTM and linear regression. *SHS Web of Conferences*.
- Andersen, Bollerslev, Diebold & Vega (2003). Micro effects of macro announcements: real-time price discovery in foreign exchange. *American Economic Review* 93(1).
- Araci (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063
- *TODO: add papers backing Gaps 1–3 (event fusion, gold event studies, dynamic/event-driven GNNs).*

