"""Leakage tests (required before any model result is reported).

1. Causality: features for day t are identical when every row after t is deleted.
2. Train-only fitting: scalers, regime thresholds, HMM (and later expected reactions / PCA) do not
   change when val/test data are changed, and were fitted on train dates only.
3. No overlap between the target periods of train, val and test.
4. Shuffled-label test: baseline_lstm trained on permuted train labels is no better than the
   majority-class rate on test.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.features.regime import build_regimes, forward_filter, fit_hmm
from src.training.dataset import build_dataset
from src.training.splits import split_of
from src.utils import paths
from src.utils.config import load_config

CFG = load_config("rcdgnn_daily")


@pytest.fixture(scope="module")
def daily_ext():
    return pd.read_parquet(paths.INTERIM / "daily_ext.parquet")


@pytest.fixture(scope="module")
def full(daily_ext):
    return build_dataset(CFG, daily_ext)


@pytest.mark.parametrize("cut", ["2020-06-15", "2022-03-16", "2024-11-05"])
def test_features_unchanged_when_future_deleted(daily_ext, full, cut):
    """Rows after `cut` removed -> every feature for days <= cut must be identical."""
    trunc = build_dataset(CFG, daily_ext.loc[:cut])
    n = len(trunc.meta)
    assert (full.meta["date"].iloc[:n].values == trunc.meta["date"].values).all()
    for name in ["X", "S", "E", "G"]:
        np.testing.assert_allclose(getattr(full, name)[:n], getattr(trunc, name), rtol=0, atol=1e-6,
                                   err_msg=f"{name} changed when future rows were deleted (cut {cut})")


def test_regime_and_hmm_causal(daily_ext):
    """Rolling-vol regimes and the filtered HMM state for day t do not use days after t."""
    reg_full, _ = build_regimes(CFG, daily_ext)
    cut = "2021-06-30"
    reg_cut, _ = build_regimes(CFG, daily_ext.loc[:cut])
    a, b = reg_full.loc[:cut], reg_cut
    assert (a["regime_vol"] == b["regime_vol"]).all()
    assert (a["regime_hmm"] == b["regime_hmm"]).all()
    # the forward filter on a prefix equals the prefix of the forward filter on the whole series
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    m = fit_hmm(X[:200], CFG)
    np.testing.assert_allclose(forward_filter(m, X)[:150], forward_filter(m, X[:150]), atol=1e-10)


def _perturb_after_train(df: pd.DataFrame) -> pd.DataFrame:
    """Scale every non-train (val/test) row's numeric data by a large factor."""
    out = df.copy()
    late = out.index > pd.Timestamp(CFG["split"]["train_end"])
    cols = ["gold_ret", "usd_ret", "bonds_dchg", "eq_ret", "oil_ret", "vix", "vix_chg"]
    out.loc[late, cols] = out.loc[late, cols] * 7.0 + 3.0
    return out


def test_scalers_and_regimes_fitted_on_train_only(daily_ext, full):
    """Changing val/test data must not change any train-period feature, threshold or HMM output."""
    pert = _perturb_after_train(daily_ext)
    d2 = build_dataset(CFG, pert)
    tr = (full.meta["split"] == "train").values
    for name in ["X", "G"]:
        np.testing.assert_allclose(getattr(full, name)[tr], getattr(d2, name)[tr], atol=1e-6)
    r1, m1 = build_regimes(CFG, daily_ext)
    r2, m2 = build_regimes(CFG, pert)
    assert m1["vol_thresholds"] == m2["vol_thresholds"]
    assert m1["hmm_scaler_mean"] == m2["hmm_scaler_mean"]
    tr_dates = r1.index <= pd.Timestamp(CFG["split"]["train_end"])
    assert (r1.loc[tr_dates, "regime_hmm"] == r2.loc[tr_dates, "regime_hmm"]).all()
    meta = json.loads((paths.PROCESSED / "regime_meta.json").read_text())
    assert meta["hmm_fit_dates"][1] <= CFG["split"]["train_end"]


def test_split_target_periods_do_not_overlap(full):
    """Target period of day t is (close_t, close_{t+1}]; splits must be strictly ordered."""
    m = full.meta.copy()
    days = pd.read_parquet(paths.INTERIM / "daily_ext.parquet").index
    nxt = pd.Series(days[1:], index=days[:-1])
    m["t_end"] = m["date"].map(nxt)
    spans = m.groupby("split").agg(start=("date", "min"), end=("t_end", "max"))
    assert spans.loc["train", "end"] <= spans.loc["val", "start"]
    assert spans.loc["val", "end"] <= spans.loc["test", "start"]
    assert set(m["split"]) == {"train", "val", "test"}
    assert (split_of(m["date"], CFG) == m["split"].values).all()


def test_shuffled_labels_give_no_edge():
    """Train baseline_lstm on permuted train labels: test accuracy ~ majority-class rate."""
    from src.training.train import mc_predict, to_batch, train_neural
    cfg = load_config("baseline_lstm")
    d = build_dataset(cfg)
    split = d.meta["split"].values
    tr, va, te = (np.where(split == s)[0] for s in ["train", "val", "test"])
    rng = np.random.default_rng(123)
    accs = []
    for seed in [0, 1]:
        model, _ = train_neural(cfg, d, tr, va, seed, y_train_override=rng.permutation(d.y[tr]))
        p, _ = mc_predict(model, to_batch(d, te), 10)
        accs.append(((p > 0.5) == d.y[te]).mean())
    majority = max(d.y[te].mean(), 1 - d.y[te].mean())
    print(f"shuffled-label test acc {np.round(accs, 3)} vs majority rate {majority:.3f}")
    assert np.mean(accs) <= majority + 0.03
