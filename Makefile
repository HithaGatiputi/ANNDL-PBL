# Usage: make all | data | features | train VARIANT=rcdgnn_daily | ablation | evaluate | dashboard | test
PY := .venv/bin/python
VARIANT ?= rcdgnn_daily
VARIANTS := baseline_majority baseline_logreg baseline_lstm baseline_lstm_news static_gnn rcdgnn_daily rcdgnn_full

.PHONY: all data features train ablation evaluate dashboard test

all: data features ablation evaluate test

data:
	$(PY) -m src.data.download_daily
	$(PY) -m src.data.download_intraday --gold-hourly
	$(PY) -m src.data.align
	$(PY) -m src.data.download_events
	$(PY) -m src.data.download_intraday --events

features:
	$(PY) -m src.features.regime
	$(PY) -m src.features.reaction_signature --mode daily
	$(PY) -m src.features.reaction_signature --mode intraday
	$(PY) -m src.models.event_encoder

train:
	$(PY) -m src.training.train --config $(VARIANT)

ablation:
	for v in $(VARIANTS); do $(PY) -m src.training.train --config $$v || exit 1; done

evaluate:
	$(PY) -m src.evaluation.ablation
	$(PY) -m src.evaluation.event_windows
	$(PY) -m src.evaluation.backtest
	$(PY) -m src.explain.graph_viz
	$(PY) -m src.explain.shap_explain
	$(PY) -m src.evaluation.summary

dashboard:
	.venv/bin/streamlit run app/dashboard.py

test:
	$(PY) -m pytest -q
