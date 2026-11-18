"""Feature importance of the reaction-signature inputs: permutation importance on the VAL set.

For each signature feature j, the column S[:, :, j] is shuffled across validation samples
(same shuffle for every day of the window) and the increase in validation log loss of the
seed-averaged model is recorded (mean over repeats). The event embedding and sentiment blocks
are permuted as groups. Positive = the model relies on it. (SHAP DeepExplainer is optional in the
plan and was skipped: permutation importance is the main method.)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.evaluation.common import available, forward_days, load_models
from src.utils import paths

REPEATS = 5


def main() -> None:
    """Write permutation_importance.csv and figure."""
    variant = available(["rcdgnn_full", "rcdgnn_daily"])[0]
    _, d, models = load_models(variant)
    va = np.where(d.meta["split"].values == "val")[0]
    y = d.y[va]
    base = log_loss(y, forward_days(models, d, va)["p_up"], labels=[0, 1])
    rng = np.random.default_rng(0)
    groups = {c: ("S", [j]) for j, c in enumerate(d.s_cols)}
    groups["event_embedding(16)"] = ("E", [j for j, c in enumerate(d.e_cols) if c.startswith("emb_")])
    groups["finbert_sentiment(3)"] = ("E", [j for j, c in enumerate(d.e_cols) if c.startswith("sent_")])
    rows = []
    for name, (arr, cols) in groups.items():
        orig = getattr(d, arr)
        inc = []
        for _ in range(REPEATS):
            perm = rng.permutation(len(va))
            new = orig.copy()
            new[np.ix_(va, np.arange(new.shape[1]), cols)] = orig[np.ix_(va[perm], np.arange(new.shape[1]), cols)]
            setattr(d, arr, new)
            inc.append(log_loss(y, forward_days(models, d, va)["p_up"], labels=[0, 1]) - base)
            setattr(d, arr, orig)
        rows.append({"feature": name, "delta_logloss_mean": np.mean(inc), "delta_logloss_std": np.std(inc)})
    tab = pd.DataFrame(rows).sort_values("delta_logloss_mean", ascending=False)
    tab.to_csv(paths.RESULTS / "permutation_importance.csv", index=False)
    top = tab.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["delta_logloss_mean"], xerr=top["delta_logloss_std"], color="C0")
    ax.set_xlabel("increase in val log loss when permuted"); ax.set_title(f"{variant}: permutation importance (val)")
    plt.tight_layout(); plt.savefig(paths.FIGURES / "permutation_importance.png", dpi=110)
    print(f"base val logloss {base:.4f}"); print(tab.head(8).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
