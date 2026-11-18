"""Event analysis: post-event days by event type, stress periods, and calibration diagrams."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.ablation import VARIANTS
from src.evaluation.common import available, seed_preds
from src.evaluation.metrics import compute_metrics
from src.utils import paths

STRESS = {"covid_2020-02_04 (val)": ("2020-02-01", "2020-04-30", "val"),
          "hikes_2022-03_12 (test)": ("2022-03-01", "2022-12-31", "test")}


def post_event_types(dates: pd.Series) -> pd.Series:
    """For each date, the type of the event on the previous trading day ('' if none)."""
    ev = pd.read_csv(paths.EVENTS_CSV)
    days = pd.read_parquet(paths.INTERIM / "daily_ext.parquet").index
    prev = pd.Series(days[:-1], index=days[1:])
    by_day = ev.groupby("trading_day")["type"].agg(lambda s: "+".join(sorted(set(s))))
    by_day.index = pd.to_datetime(by_day.index)
    return dates.map(prev).map(by_day).fillna("")


def mean_std(frames: list[pd.DataFrame], mask_fn) -> dict:
    """Metrics per seed on a subset, then mean ± std."""
    ms = []
    for f in frames:
        m = mask_fn(f)
        ms.append(compute_metrics(f["y"].values[m], f["p_up"].values[m]))
    out = {"n": ms[0]["n"]}
    for k in ["acc", "mcc", "brier", "logloss"]:
        v = np.array([x[k] for x in ms])
        out[k] = f"{np.nanmean(v):.3f} ± {np.nanstd(v):.3f}"
    return out


def main() -> None:
    """Write event_analysis.csv/.md and calibration.png."""
    rows = []
    for v in available(VARIANTS):
        test = seed_preds(v, "test")
        types = post_event_types(test[0]["date"])
        for t in ["FOMC", "CPI", "NFP"]:
            rows.append({"variant": v, "subset": f"post_event_{t}",
                         **mean_std(test, lambda f, t=t: types.str.contains(t).values)})
        for name, (a, b, split) in STRESS.items():
            frames = seed_preds(v, split)
            rows.append({"variant": v, "subset": name,
                         **mean_std(frames, lambda f: ((f["date"] >= a) & (f["date"] <= b)).values)})
    tab = pd.DataFrame(rows)
    tab.to_csv(paths.RESULTS / "event_analysis.csv", index=False)
    (paths.RESULTS / "event_analysis.md").write_text(
        "# Event analysis (mean ± std over seeds)\n\nPost-event = the trading day after an event of that type "
        "(test set). Stress periods: COVID Feb–Apr 2020 lies in the validation split; 2022 hikes in test.\n\n"
        + tab.to_markdown(index=False))
    print(tab[tab["variant"].isin(["static_gnn", "rcdgnn_daily", "rcdgnn_full"])].to_string(index=False))

    # reliability diagram (seed-ensembled test predictions)
    vs = available(VARIANTS)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    bins = np.linspace(0, 1, 11)
    for v in vs:
        fr = seed_preds(v, "test")
        p = np.mean([f["p_up"].values for f in fr], 0)
        y = fr[0]["y"].values
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 10]
        ax.plot(xs, ys, "o-", ms=3, label=v)
    ax.set_xlim(0.3, 0.7); ax.set_ylim(0.3, 0.7)
    ax.set_xlabel("predicted P(up) (bins with >= 10 days)"); ax.set_ylabel("observed up-rate")
    ax.set_title("Reliability diagram, test set (seed ensemble)"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(paths.FIGURES / "calibration.png", dpi=110)


if __name__ == "__main__":
    main()
