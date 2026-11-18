"""Feature unit tests. The price paths here are SYNTHETIC (made up for testing only)."""
import numpy as np
import pandas as pd

from src.features.reaction_signature import compute_signature
from src.features.technical import rsi


def _synthetic_path():
    """SYNTHETIC 1-minute path: 60 quiet minutes (+/-sigma alternating), then a +10 sigma jump
    3 minutes after the event that is held until +1h, then half of it reverses by the close."""
    sigma = 0.0001
    t0 = pd.Timestamp("2021-01-01 14:00", tz="UTC")
    pre = np.cumsum(np.tile([sigma, -sigma], 30))                    # 60 pre-event minutes
    post = np.zeros(180)
    post[2:] = 10 * sigma                                            # bar ending at t0+3min jumps
    post[90:] = 5 * sigma                                            # half reverses after 1.5h
    logp = np.r_[pre - pre[-1], post]                                # price at event = 1.0
    idx = t0 + pd.to_timedelta(np.arange(-59, 181), unit="min")      # bar END times
    return pd.Series(np.exp(logp), index=idx), t0, sigma


def test_signature_on_synthetic_path():
    path, t0, sigma = _synthetic_path()
    pre_ret = np.log(path[path.index <= t0]).diff().dropna()
    sd = pre_ret.std()
    sig = compute_signature({"gold": path}, t0, {"gold": sd}, ["5m", "1h", "close"], {"mag_close": 0.0})
    assert sig["latency_min"] == 3.0
    assert sig["dir_5m"] == 1 and sig["dir_1h"] == 1 and sig["dir_close"] == 1
    np.testing.assert_allclose(sig["mag_5m"], 10 * sigma / (sd * np.sqrt(5)), rtol=1e-6)
    np.testing.assert_allclose(sig["mag_1h"], 10 * sigma / (sd * np.sqrt(60)), rtol=1e-6)
    np.testing.assert_allclose(sig["persistence"], 0.5, rtol=1e-6)
    assert sig["has_intraday"] == 1.0
    assert sig["vol_ratio"] > 0


def test_signature_daily_proxy():
    """SYNTHETIC two-point daily path: magnitude = return / daily vol, no intraday fields."""
    t_prev = pd.Timestamp("2021-01-04 22:00", tz="UTC")
    t_now = pd.Timestamp("2021-01-05 22:00", tz="UTC")
    p = pd.Series([1.0, np.exp(0.02)], index=[t_prev, t_now])
    sig = compute_signature({"gold": p}, t_prev + pd.Timedelta(seconds=1), {"gold": 0.01}, ["close"], {"mag_close": 0.5})
    np.testing.assert_allclose(sig["mag_close"], 2.0)
    np.testing.assert_allclose(sig["vol_ratio"], 2.0)
    np.testing.assert_allclose(sig["gap_gold"], 1.5)
    assert np.isnan(sig["latency_min"]) and np.isnan(sig["persistence"]) and sig["has_intraday"] == 0


def test_rsi_bounds():
    """RSI of a SYNTHETIC random walk stays in [0, 100]."""
    s = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 300))))
    r = rsi(s).dropna()
    assert ((r >= 0) & (r <= 100)).all()
