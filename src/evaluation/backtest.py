"""SIMPLIFIED long/flat backtest. NOT evidence of tradable profit.

Long gold from the 17:00 ET close of day t to the close of t+1 when P(up) > threshold, flat
otherwise; cost = cost_bp per position change. Seed-ensembled test predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.ablation import VARIANTS
from src.evaluation.common import available, seed_preds
from src.utils import paths
from src.utils.config import load_config


def stats(r: np.ndarray) -> dict:
    """Annualised Sharpe (252 days), total log return, max drawdown."""
    eq = np.cumsum(r)
    dd = eq - np.maximum.accumulate(eq)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    return {"sharpe": sh, "total_log_ret": eq[-1], "max_drawdown": dd.min()}


def main() -> None:
    """Write backtest.md."""
    cfg = load_config("base")
    th, cost = cfg["eval"]["backtest_threshold"], cfg["eval"]["backtest_cost_bp"] / 1e4
    daily = pd.read_parquet(paths.INTERIM / "daily_ext.parquet")
    nxt = np.log(daily["gold_close"].shift(-1) / daily["gold_close"])  # return t -> t+1 (the target period)
    rows = []
    for v in available(VARIANTS):
        fr = seed_preds(v, "test")
        p = np.mean([f["p_up"].values for f in fr], 0)
        r_next = nxt.reindex(fr[0]["date"]).values
        pos = (p > th).astype(float)
        trades = np.abs(np.diff(np.r_[0.0, pos]))
        strat = pos * r_next - trades * cost
        hit = (r_next[pos == 1] > 0).mean() if pos.sum() else np.nan
        rows.append({"variant": v, "days_long": int(pos.sum()), "trades": int(trades.sum()), "hit_rate": hit,
                     **stats(strat)})
    bh = stats(nxt.reindex(fr[0]["date"]).values)
    rows.append({"variant": "buy_and_hold", "days_long": len(fr[0]), "trades": 1, "hit_rate": np.nan, **bh})
    tab = pd.DataFrame(rows)
    (paths.RESULTS / "backtest.md").write_text(
        "# Simplified backtest (test set) — NOT evidence of tradable profit\n\n"
        f"Long when P(up) > {th}, flat otherwise; {cfg['eval']['backtest_cost_bp']} bp per position change; "
        "no slippage, financing or spread modelling. The 2022–2026 gold rally makes buy-and-hold hard to beat.\n\n"
        + tab.round(4).to_markdown(index=False))
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
