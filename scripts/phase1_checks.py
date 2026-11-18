"""Phase 1 acceptance checks: row count, NaNs, class balance per split, correlation, sanity plot."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.splits import split_of
from src.utils import paths
from src.utils.config import load_config

cfg = load_config("base")
df = pd.read_parquet(paths.DAILY_PARQUET)
print(f"rows={len(df)} range={df.index.min().date()}..{df.index.max().date()} NaNs={int(df.isna().sum().sum())}")
df["split"] = split_of(df.index, cfg)
print(df.groupby("split")["y_next"].agg(["count", "mean"]).round(3).to_string())
print(f"corr(gold_ret, usd_ret) = {df['gold_ret'].corr(df['usd_ret']):.3f}")
lv = pd.DataFrame({"gold": df["gold_close"],
                   "usd": np.exp(df["usd_ret"].cumsum()),
                   "bonds_10y_yield_level_chg_bp": df["bonds_dchg"].cumsum(),
                   "equities": np.exp(df["eq_ret"].cumsum()),
                   "oil": np.exp(df["oil_ret"].cumsum())})
norm = (lv - lv.mean()) / lv.std()
ax = norm.plot(figsize=(11, 5), lw=0.8, title="Phase 1: normalised series (z-score of level)")
for d in [cfg["split"]["train_end"], cfg["split"]["val_end"]]:
    ax.axvline(pd.Timestamp(d), color="k", ls="--", lw=0.7)
plt.tight_layout(); plt.savefig(paths.FIGURES / "phase1_prices.png", dpi=110)
print("saved phase1_prices.png")
