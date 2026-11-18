"""Volatility regimes (main) and a 3-state Gaussian HMM (check).

regime_vol: 20-day rolling std of gold_ret vs TRAIN-split percentiles
            (<33 calm, 33-67 normal, >67 volatile, >95 crisis; crisis overrides volatile).
regime_hmm: GaussianHMM(3) fitted on train [gold_ret, rolling_vol]; states are decoded with the
            FORWARD (filtering) recursion only, so the label for day t never uses days after t.
            States are mapped to calm/normal/volatile by their mean volatility.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from src.training.splits import split_of
from src.utils import paths
from src.utils.config import load_config

LABELS = ["calm", "normal", "volatile", "crisis"]


def rolling_vol(ret: pd.Series, window: int) -> pd.Series:
    """Rolling std of returns over the last `window` days (including day t)."""
    return ret.rolling(window).std()


def vol_thresholds(vol: pd.Series, train_mask: np.ndarray, cfg: dict) -> dict[str, float]:
    """Percentile thresholds computed on train days only."""
    v = vol[train_mask].dropna()
    r = cfg["regime"]
    return {k: float(np.percentile(v, r[f"pct_{k}"])) for k in ["calm", "volatile", "crisis"]}


def label_vol(vol: pd.Series, th: dict[str, float]) -> pd.Series:
    """Map volatility to calm / normal / volatile / crisis."""
    lab = pd.Series("normal", index=vol.index)
    lab[vol < th["calm"]] = "calm"
    lab[vol > th["volatile"]] = "volatile"
    lab[vol > th["crisis"]] = "crisis"
    return lab


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Filtered state probabilities P(state_t | x_1..x_t) using only past and current data."""
    n, k = len(X), model.n_components
    covs = model.covars_  # full (k, d, d) matrices
    log_b = np.column_stack([multivariate_normal(model.means_[j], covs[j]).logpdf(X) for j in range(k)])
    log_a = np.log(model.transmat_)
    log_alpha = np.zeros((n, k))
    log_alpha[0] = np.log(model.startprob_) + log_b[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for t in range(1, n):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
        log_alpha[t] -= logsumexp(log_alpha[t])
    return np.exp(log_alpha)


def fit_hmm(X_train: np.ndarray, cfg: dict) -> GaussianHMM:
    """Fit the HMM on standardised train features."""
    r = cfg["regime"]
    m = GaussianHMM(n_components=r["hmm_states"], covariance_type="full", n_iter=r["hmm_iter"],
                    random_state=r["hmm_seed"])
    m.fit(X_train)
    return m


def build_regimes(cfg: dict, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Compute both regime labels on the extended daily table (warm-up rows included).

    `df` can be passed in (used by the leakage tests); by default daily_ext.parquet is read.
    """
    if df is None:
        df = pd.read_parquet(paths.INTERIM / "daily_ext.parquet")
    vol = rolling_vol(df["gold_ret"], cfg["regime"]["vol_window"])
    train = (split_of(df.index, cfg) == "train") & vol.notna().values
    th = vol_thresholds(vol, train, cfg)
    out = pd.DataFrame(index=df.index)
    out["rolling_vol"] = vol
    out["regime_vol"] = label_vol(vol, th)

    feats = np.column_stack([df["gold_ret"].values, vol.values])
    ok = ~np.isnan(feats).any(axis=1)
    mu, sd = feats[train].mean(0), feats[train].std(0)  # train-only standardisation
    Z = (feats - mu) / sd
    hmm = fit_hmm(Z[train], cfg)
    probs = np.full((len(df), hmm.n_components), np.nan)
    probs[ok] = forward_filter(hmm, Z[ok])
    order = np.argsort(hmm.means_[:, 1])          # ascending mean volatility
    name = {int(s): LABELS[i] for i, s in enumerate(order)}
    state = np.where(ok, np.nanargmax(np.nan_to_num(probs, nan=-1), axis=1), -1)
    out["regime_hmm"] = [name.get(int(s), "normal") if o else None for s, o in zip(state, ok)]
    out = out[ok]
    meta = {"vol_thresholds": th, "train_end": cfg["split"]["train_end"],
            "hmm_fit_dates": [str(df.index[train].min().date()), str(df.index[train].max().date())],
            "hmm_scaler_mean": mu.tolist(), "hmm_scaler_std": sd.tolist()}
    return out, meta


def main() -> None:
    """Build regimes, save parquet + thresholds, print checks and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import cohen_kappa_score

    cfg = load_config("base")
    reg, meta = build_regimes(cfg)
    reg.to_parquet(paths.REGIMES_PARQUET)
    (paths.PROCESSED / "regime_meta.json").write_text(json.dumps(meta, indent=2))
    period = reg.loc[cfg["data"]["start"]:cfg["data"]["end"]]
    print("regime_vol counts:", period["regime_vol"].value_counts().to_dict())
    print("regime_hmm counts:", period["regime_hmm"].value_counts().to_dict())
    merged = period["regime_vol"].replace("crisis", "volatile")
    print(f"Cohen's kappa (vol[crisis->volatile] vs hmm): {cohen_kappa_score(merged, period['regime_hmm']):.3f}")
    m20 = period.loc["2020-03"]
    for c in ["regime_vol", "regime_hmm"]:
        frac = m20[c].isin(["volatile", "crisis"]).mean()
        print(f"March 2020 {c}: {frac:.0%} volatile/crisis")
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(period.index, period["rolling_vol"], lw=0.7, color="k")
    for t in ["calm", "volatile", "crisis"]:
        ax[0].axhline(meta["vol_thresholds"][t], ls="--", lw=0.6, label=f"{t} thr (train)")
    ax[0].legend(fontsize=7); ax[0].set_title("20-day gold volatility and regimes")
    codes = {"calm": 0, "normal": 1, "volatile": 2, "crisis": 3}
    ax[1].plot(period.index, period["regime_vol"].map(codes), lw=0.6, label="vol percentile")
    ax[1].plot(period.index, period["regime_hmm"].map(codes) + 0.1, lw=0.6, label="HMM (filtered)")
    ax[1].set_yticks(range(4)); ax[1].set_yticklabels(list(codes)); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(paths.FIGURES / "phase2_regimes.png", dpi=110)


if __name__ == "__main__":
    main()
