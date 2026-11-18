"""Ablation table + significance tests over all variants.

ablation_table.csv / .md : variant x subset {all, event, post_event, calm, volatile, crisis}
                           x metric {acc, f1, mcc, brier, logloss, ece}, mean ± std over seeds (test set).
significance.csv / .md   : McNemar (seed-ensembled, threshold 0.5) and Diebold–Mariano on Brier loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from src.evaluation.metrics import METRICS
from src.utils import paths

VARIANTS = ["baseline_majority", "baseline_logreg", "baseline_lstm", "baseline_lstm_news",
            "static_gnn", "rcdgnn_daily", "rcdgnn_full"]
SUBSETS = ["all", "event", "post_event", "calm", "volatile", "crisis"]
PAIRS = [("static_gnn", "rcdgnn_daily"), ("rcdgnn_daily", "rcdgnn_full"), ("baseline_lstm", "rcdgnn_full")]


def seed_metrics(variant: str, split: str = "test") -> list[dict]:
    """metrics.json of every seed of a variant."""
    files = sorted((paths.RESULTS / variant).glob("seed*/metrics.json"))
    return [json.loads(f.read_text())[split] for f in files]


def results_table(variants: list[str], subsets=SUBSETS, metrics=METRICS, split: str = "test") -> pd.DataFrame:
    """Long table with mean/std over seeds for each variant x subset x metric."""
    rows = []
    for v in variants:
        ms = seed_metrics(v, split)
        if not ms:
            continue
        for s in subsets:
            row = {"variant": v, "subset": s, "n_days": ms[0][s]["n"], "n_seeds": len(ms)}
            for m in metrics:
                vals = np.array([x[s][m] for x in ms], dtype=float)
                row[f"{m}_mean"], row[f"{m}_std"] = np.nanmean(vals), np.nanstd(vals)
            rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(tab: pd.DataFrame, metrics=METRICS) -> str:
    """Compact markdown: one cell 'mean ± std' per metric."""
    out = tab[["variant", "subset", "n_days", "n_seeds"]].copy()
    for m in metrics:
        out[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(tab[f"{m}_mean"], tab[f"{m}_std"])]
    return out.to_markdown(index=False)


def ensemble_preds(variant: str, split: str = "test") -> pd.DataFrame:
    """Average p_up over seeds (seed ensemble)."""
    files = sorted((paths.RESULTS / variant).glob(f"seed*/preds_{split}.csv"))
    dfs = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    out = dfs[0].copy()
    out["p_up"] = np.mean([d["p_up"].values for d in dfs], axis=0)
    return out


def diebold_mariano(l1: np.ndarray, l2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test on loss differential d = l1 - l2 (h=1, so no autocorrelation terms). Two-sided p."""
    d = l1 - l2
    n = len(d)
    var = np.var(d, ddof=0) / n
    dm = d.mean() / np.sqrt(var)
    # Harvey-Leybourne-Newbold small-sample correction, t distribution with n-1 df
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(dm), df=n - 1)
    return float(dm), float(p)


def significance(pairs=PAIRS, subset: str = "all") -> pd.DataFrame:
    """McNemar + DM for each pair on test days (optionally post-event days only)."""
    rows = []
    for a, b in pairs:
        if not (paths.RESULTS / a).exists() or not (paths.RESULTS / b).exists():
            continue
        pa, pb = ensemble_preds(a), ensemble_preds(b)
        assert (pa["date"].values == pb["date"].values).all()
        m = np.ones(len(pa), bool) if subset == "all" else pa["is_post_event_day"].values.astype(bool)
        y = pa["y"].values[m]
        ca, cb = (pa["p_up"].values[m] > 0.5) == y, (pb["p_up"].values[m] > 0.5) == y
        table = [[np.sum(ca & cb), np.sum(ca & ~cb)], [np.sum(~ca & cb), np.sum(~ca & ~cb)]]
        mc = mcnemar(table, exact=True)
        la, lb = (pa["p_up"].values[m] - y) ** 2, (pb["p_up"].values[m] - y) ** 2
        dm, dmp = diebold_mariano(la, lb)
        rows.append({"model_a": a, "model_b": b, "subset": subset, "n": int(m.sum()),
                     "acc_a": ca.mean(), "acc_b": cb.mean(), "a_right_b_wrong": table[0][1],
                     "a_wrong_b_right": table[1][0], "mcnemar_p": float(mc.pvalue),
                     "brier_a": la.mean(), "brier_b": lb.mean(), "dm_stat": dm, "dm_p": dmp,
                     "significant_5pct": bool(min(mc.pvalue, dmp) < 0.05)})
    return pd.DataFrame(rows)


def main() -> None:
    """Write the ablation and significance tables."""
    tab = results_table(VARIANTS)
    tab.to_csv(paths.RESULTS / "ablation_table.csv", index=False)
    md = "# Ablation table (test set, mean ± std over seeds)\n\n" + to_markdown(tab)
    vt = results_table(VARIANTS, subsets=["all", "post_event"], split="val")
    vt.to_csv(paths.RESULTS / "ablation_table_val.csv", index=False)
    md += "\n\n# Validation set\n\n" + to_markdown(vt)
    (paths.RESULTS / "ablation_table.md").write_text(md)
    sig = pd.concat([significance(subset="all"), significance(subset="post_event")], ignore_index=True)
    sig.to_csv(paths.RESULTS / "significance.csv", index=False)
    (paths.RESULTS / "significance.md").write_text(
        "# Significance tests (test set, seed-ensembled predictions)\n\n"
        "McNemar: exact binomial test on disagreements at threshold 0.5. "
        "DM: Diebold–Mariano (HLN-corrected) on squared-error (Brier) loss; negative stat = model_a better.\n\n"
        + sig.round(4).to_markdown(index=False))
    print(to_markdown(tab[tab["subset"].isin(["all", "post_event"])], metrics=["acc", "mcc", "brier"]))
    print(sig.round(4).to_string())


if __name__ == "__main__":
    main()
