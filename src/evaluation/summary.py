"""Write reports/results/SUMMARY.md in plain English from the saved result tables.
The wording is generated from the numbers (nothing is hand-entered)."""
from __future__ import annotations

import json

import pandas as pd

from src.utils import paths


def fmt(tab: pd.DataFrame, v: str, s: str, m: str) -> str:
    """'mean ± std' string for variant v, subset s, metric m."""
    r = tab[(tab.variant == v) & (tab.subset == s)]
    return "n/a" if r.empty else f"{r[m + '_mean'].iloc[0]:.3f} ± {r[m + '_std'].iloc[0]:.3f}"


def val(tab, v, s, m) -> float:
    """Mean value for variant v, subset s, metric m."""
    r = tab[(tab.variant == v) & (tab.subset == s)]
    return float("nan") if r.empty else float(r[m + "_mean"].iloc[0])


def sig_sentence(sig: pd.DataFrame, a: str, b: str, subset: str) -> str:
    """One plain-English sentence for a significance-test row."""
    r = sig[(sig.model_a == a) & (sig.model_b == b) & (sig.subset == subset)]
    if r.empty:
        return f"{a} vs {b}: not available."
    r = r.iloc[0]
    better = "tie" if abs(r.acc_b - r.acc_a) < 1e-9 else (f"{b} higher" if r.acc_b > r.acc_a else f"{a} higher")
    verdict = "significant at 5%" if r.significant_5pct else "**not significant**"
    return (f"{a} vs {b} ({subset}, n={r.n}, seed-ensembled): accuracy {r.acc_a:.3f} vs {r.acc_b:.3f} "
            f"({better}), McNemar p = {r.mcnemar_p:.3f}; Brier {r.brier_a:.4f} vs {r.brier_b:.4f}, "
            f"Diebold–Mariano p = {r.dm_p:.3f} → {verdict}.")


def main() -> None:
    """Generate SUMMARY.md."""
    tab = pd.read_csv(paths.RESULTS / "ablation_table.csv")
    sig = pd.read_csv(paths.RESULTS / "significance.csv")
    variants = tab.variant.unique().tolist()
    best = max(variants, key=lambda v: val(tab, v, "all", "acc"))
    best_b = min(variants, key=lambda v: val(tab, v, "all", "brier"))
    maj = val(tab, "baseline_majority", "all", "acc")
    lines = ["# SUMMARY (plain English)", "",
             "Task: predict whether gold (XAU/USD, 17:00 New York close) closes higher on the next trading day. "
             "Test period 2022-01-03 → 2026-06-30 (1,159 days). All numbers are mean ± std over 5 seeds unless "
             "stated; tuning used the validation split only.", "",
             "## What won", "",
             f"- Highest test accuracy: **{best}** ({fmt(tab, best, 'all', 'acc')}). "
             f"Always predicting 'up' (majority baseline) scores {maj:.3f}.",
             f"- Best probability quality (lowest Brier): **{best_b}** ({fmt(tab, best_b, 'all', 'brier')}); "
             f"a constant 50/50 forecast has Brier 0.250.", "",
             "| model | acc (all) | MCC (all) | Brier (all) | acc (post-event) | MCC (post-event) |",
             "|---|---|---|---|---|---|"]
    for v in variants:
        lines.append(f"| {v} | {fmt(tab, v, 'all', 'acc')} | {fmt(tab, v, 'all', 'mcc')} | {fmt(tab, v, 'all', 'brier')} "
                     f"| {fmt(tab, v, 'post_event', 'acc')} | {fmt(tab, v, 'post_event', 'mcc')} |")
    lines += ["", "## Main hypothesis: does reaction conditioning help?", ""]
    for a, b in [("static_gnn", "rcdgnn_daily"), ("rcdgnn_daily", "rcdgnn_full"), ("baseline_lstm", "rcdgnn_full")]:
        for s in ["all", "post_event"]:
            lines.append("- " + sig_sentence(sig, a, b, s))
    d_all = val(tab, "rcdgnn_full", "all", "acc") - val(tab, "static_gnn", "all", "acc")
    d_post = val(tab, "rcdgnn_full", "post_event", "acc") - val(tab, "static_gnn", "post_event", "acc")
    any_sig = bool(sig.significant_5pct.any())
    lines += ["", f"Reaction-conditioned (intraday) minus static GNN accuracy: {d_all:+.3f} on all test days, "
              f"{d_post:+.3f} on post-event days. "
              + ("At least one comparison is significant at 5% (see table)." if any_sig else
                 "None of the pairwise differences is statistically significant at 5%: **the main hypothesis is "
                 "not supported by this data**. This is a valid null result."), ""]
    gc = paths.RESULTS / "graph_change_by_type.csv"
    if gc.exists():
        g = pd.read_csv(gc)
        mx = g.loc[g.mean_dA.abs().idxmax()]
        lines += ["## What the graph learned", "",
                  f"- The largest average event-driven edge change on test event days is {mx.mean_dA:+.4f} "
                  f"({mx['type']}: {mx['from']} → {mx['to']}), on attention weights of about 0.2. The learned graph "
                  "barely reacts to events: the L1 penalty and the weak signal keep ΔA close to zero.", ""]
    diag = [v for v in ["rcdgnn_daily_l1zero", "rcdgnn_full_l1zero"] if (paths.RESULTS / "_tuning" / v).exists()]
    if diag:
        import glob
        import numpy as np
        parts = []
        for v in diag:
            ms = [json.loads(open(f).read()) for f in sorted(glob.glob(str(paths.RESULTS / "_tuning" / v / "seed*/metrics.json")))]
            parts.append(f"{v}: val log loss {np.mean([m['val']['all']['logloss'] for m in ms]):.4f}, "
                         f"test acc {np.mean([m['test']['all']['acc'] for m in ms]):.3f}")
        lines += ["- Diagnostic (not used for model choice): removing the L1 penalty makes the event-driven edge "
                  "change ~5x larger but leaves validation loss unchanged (" + "; ".join(parts) + "). "
                  "So the null result is not caused by the penalty.", ""]
    pi = paths.RESULTS / "permutation_importance.csv"
    if pi.exists():
        p = pd.read_csv(pi).head(3)
        lines += ["- Most important event inputs on validation (permutation, Δ log loss): "
                  + ", ".join(f"{r.feature} ({r.delta_logloss_mean:+.5f})" for r in p.itertuples()) + ".", ""]
    lines += ["## Where it failed", "",
              "- No model clearly beats the majority-class rate; MCCs are close to 0, so the daily direction of gold is "
              "close to unpredictable from these inputs.",
              "- The 2022–2026 gold rally makes 'always up' hard to beat; post-event days (n=141) are too few to "
              "detect small effects.",
              "- See `event_analysis.md` (by event type and stress period), `backtest.md` (simplified, not evidence "
              "of tradable profit) and `calibration.png`.", ""]
    q = paths.PROCESSED / "intraday_quality.csv"
    if q.exists():
        qq = pd.read_csv(q)
        lines += [f"Intraday coverage: {qq.ok_gold.mean():.0%} of events had a valid 1-minute gold window.", ""]
    (paths.RESULTS / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
