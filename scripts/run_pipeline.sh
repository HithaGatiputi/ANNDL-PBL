#!/usr/bin/env bash
set -e
python -m src.data.download_daily
python -m src.data.download_intraday
python -m src.data.download_events
python -m src.data.download_news
python -m src.data.download_intraday
python -m src.data.align
python -m src.training.train --config configs/baseline_lstm.yaml
python -m src.training.train --config configs/rc_dgnn_daily.yaml
python -m src.evaluation.ablation
