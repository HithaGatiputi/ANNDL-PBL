"""Graph explainability: how the learned 5x5 adjacency changes on event days.

For the main reaction-conditioned model (seed-averaged, dropout off):
  * graph_change_<event_id>.png : A on the day before, A on the event day, and the change caused
    by the event (A_event - A with the event edges switched off) for a few major events.
  * graph_change_by_type.png/.csv : mean event-driven change per event type (test events).
  * data/processed/graph_<variant>.npz : A and the no-event A for every modelling day (dashboard).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.common import available, forward_days, load_models
from src.utils import paths

NODES = ["Gold", "USD", "Bonds", "Equities", "Oil"]


def heat(ax, A, title, diff=False):
    """One annotated heatmap."""
    vmax = np.abs(A).max() if diff else 1.0
    im = ax.imshow(A, cmap="RdBu_r" if diff else "viridis", vmin=-vmax if diff else 0, vmax=vmax)
    ax.set_xticks(range(5)); ax.set_xticklabels(NODES, rotation=45, fontsize=7)
    ax.set_yticks(range(5)); ax.set_yticklabels(NODES, fontsize=7)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{A[i, j]:+.3f}" if diff else f"{A[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="k" if diff else "w")
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)


def pick_events(variant_sig: str, val_end: str) -> list[str]:
    """Two named FOMC events plus the test CPI and NFP events with the largest |reaction gap|."""
    sig = pd.read_csv(paths.PROCESSED / f"event_signatures_{variant_sig}.csv")
    test = sig[sig["trading_day"] > val_end]
    out = ["FOMC_20200315", "FOMC_20220615"]
    for t in ["CPI", "NFP"]:
        sub = test[test["type"] == t]
        out.append(sub.loc[sub["gap_gold"].abs().idxmax(), "event_id"])
    return out


def main() -> None:
    """Make all graph figures."""
    variant = available(["rcdgnn_full", "rcdgnn_daily"])[0]
    cfg, d, models = load_models(variant)
    allidx = np.arange(len(d.meta))
    full = forward_days(models, d, allidx)
    noev = forward_days(models, d, allidx, no_event_edges=True)
    dates = pd.to_datetime(d.meta["date"])
    np.savez_compressed(paths.PROCESSED / f"graph_{variant}.npz", dates=dates.dt.strftime("%Y-%m-%d").values,
                        A=full["A"], A_noevent=noev["A"], gate=full["gate"], p_up=full["p_up"])
    mode = "intraday" if variant == "rcdgnn_full" else "daily"
    ev = pd.read_csv(paths.EVENTS_CSV)
    pos = {dt: i for i, dt in enumerate(dates)}
    for eid in pick_events(mode, cfg["split"]["val_end"]):
        td = pd.Timestamp(ev.loc[ev["event_id"] == eid, "trading_day"].iloc[0])
        i = pos[td]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        heat(ax[0], full["A"][i - 1], f"A on day before ({dates[i - 1].date()})")
        heat(ax[1], full["A"][i], f"A on event day ({td.date()}), gate={full['gate'][i]:.2f}")
        heat(ax[2], full["A"][i] - noev["A"][i], "event-driven change (A - A without event edges)", diff=True)
        fig.suptitle(f"{variant}: {eid} (rows = receiving node, row-softmax weights)", fontsize=9)
        plt.tight_layout(); plt.savefig(paths.FIGURES / f"graph_change_{eid}.png", dpi=110); plt.close(fig)
        print(f"saved graph_change_{eid}.png")
    # average event-driven change per event type on test event days
    test_ev = ev[ev["trading_day"] > cfg["split"]["val_end"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    rows = []
    for k, t in enumerate(["FOMC", "CPI", "NFP"]):
        ids = [pos[pd.Timestamp(x)] for x in test_ev.loc[test_ev["type"] == t, "trading_day"] if pd.Timestamp(x) in pos]
        dA = (full["A"][ids] - noev["A"][ids]).mean(0)
        heat(ax[k], dA, f"{t}: mean change (n={len(ids)} test events)", diff=True)
        for i in range(5):
            for j in range(5):
                rows.append({"type": t, "to": NODES[i], "from": NODES[j], "mean_dA": dA[i, j], "n_events": len(ids)})
    plt.tight_layout(); plt.savefig(paths.FIGURES / "graph_change_by_type.png", dpi=110); plt.close(fig)
    pd.DataFrame(rows).to_csv(paths.RESULTS / "graph_change_by_type.csv", index=False)
    ev_days = d.meta["is_event_day"].values
    print(f"mean gate on event days {full['gate'][ev_days].mean():.3f} vs other days {full['gate'][~ev_days].mean():.3f}")


if __name__ == "__main__":
    main()
