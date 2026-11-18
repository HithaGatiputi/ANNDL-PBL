"""RC-DGNN dashboard (Streamlit). Reads only saved files in reports/ and data/; never trains.

Run: streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils import paths  # noqa: E402

NODES = ["Gold", "USD", "Bonds", "Equities", "Oil"]
MODELS = ["rcdgnn_full", "rcdgnn_daily", "static_gnn", "baseline_lstm_news", "baseline_lstm"]


@st.cache_data
def load_preds(variant: str) -> pd.DataFrame:
    """Seed-ensembled test predictions: mean p_up, mean MC std, std across seeds."""
    fr = [pd.read_csv(f, parse_dates=["date"]) for f in sorted((paths.RESULTS / variant).glob("seed*/preds_test.csv"))]
    out = fr[0][["date", "y", "regime", "is_event_day", "is_post_event_day"]].copy()
    out["p_up"] = np.mean([f["p_up"] for f in fr], 0)
    out["p_std_mc"] = np.mean([f["p_std"] for f in fr], 0)
    out["p_std_seeds"] = np.std([f["p_up"] for f in fr], 0)
    return out


@st.cache_data
def load_sig(file: str) -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED / file)[["last_event_id", "days_since_event"]]


@st.cache_data
def load_graph(variant: str):
    f = paths.PROCESSED / f"graph_{variant}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


@st.cache_data
def load_events() -> pd.DataFrame:
    return pd.read_csv(paths.EVENTS_CSV)


def draw_graph(A: np.ndarray, title: str):
    """5-node circle graph; edge width = weight (row-softmax: i receives from j)."""
    ang = np.linspace(0, 2 * np.pi, 6)[:-1] + np.pi / 2
    xy = np.c_[np.cos(ang), np.sin(ang)]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for i in range(5):
        for j in range(5):
            if i != j:
                w = (A[i, j] + A[j, i]) / 2
                ax.plot(*zip(xy[i], xy[j]), color="C0", lw=w * 25, alpha=0.6)
    ax.scatter(xy[:, 0], xy[:, 1], s=900, color="gold", zorder=3, edgecolor="k")
    for k, n in enumerate(NODES):
        ax.text(*xy[k], n, ha="center", va="center", fontsize=7, zorder=4)
    ax.set_title(title, fontsize=8); ax.axis("off"); ax.set_aspect("equal")
    return fig


st.set_page_config(page_title="RC-DGNN gold forecaster", layout="wide")
page = st.sidebar.radio("Page", ["Forecast", "Graph", "Event explorer", "Results"])
avail = [m for m in MODELS if (paths.RESULTS / m).exists()]
model = st.sidebar.selectbox("Model", avail)
st.sidebar.caption("Decision-support prototype, not a trading system.")

if page == "Forecast":
    st.title("Next-day gold direction forecast (test period)")
    df = load_preds(model)
    d = st.selectbox("Test date", df["date"].dt.date.tolist()[::-1])
    r = df[df["date"].dt.date == d].iloc[0]
    sigfile = "signatures_intraday.parquet" if (paths.SIG_INTRADAY.exists() and model == "rcdgnn_full") else "signatures_daily.parquet"
    sig = load_sig(sigfile)
    s = sig.loc[pd.Timestamp(d)]
    c = st.columns(4)
    c[0].metric("P(gold up tomorrow)", f"{r.p_up:.3f}", help="seed-ensembled MC-dropout mean")
    c[1].metric("Uncertainty (MC std)", f"± {r.p_std_mc:.3f}")
    c[2].metric("Regime", r.regime)
    c[3].metric("Outcome", "UP" if r.y == 1 else "DOWN", "correct" if (r.p_up > 0.5) == (r.y == 1) else "wrong")
    days = int(round(s.days_since_event * 10))
    st.write(f"**Last event:** {s.last_event_id or 'none in last 10 trading days'}"
             + (f" — {days} trading day(s) ago" if s.last_event_id else ""))
    win = df[(df["date"] >= pd.Timestamp(d) - pd.Timedelta(days=60)) & (df["date"] <= pd.Timestamp(d))]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.plot(win["date"], win["p_up"], color="C0", label="P(up)")
    ax.fill_between(win["date"], win["p_up"] - 2 * win["p_std_mc"], win["p_up"] + 2 * win["p_std_mc"], alpha=0.2)
    ax.scatter(win["date"], 0.5 + 0.02 * (2 * win["y"] - 1), s=8, color="k", label="actual (up/down)")
    ax.axhline(0.5, color="grey", lw=0.6); ax.legend(fontsize=7); ax.set_title("last 60 days (band = ±2 MC std)", fontsize=8)
    st.pyplot(fig)

elif page == "Graph":
    st.title("Asset graph before and after the last event")
    variant = model if model.startswith("rcdgnn") else "rcdgnn_full"
    g = load_graph(variant) or load_graph("rcdgnn_daily")
    if g is None:
        st.warning("Run `make evaluate` first.")
    else:
        dates = pd.to_datetime(g["dates"])
        test_dates = [x.date() for x in dates if x >= pd.Timestamp("2022-01-01")]
        d = st.selectbox("Date", test_dates[::-1])
        i = int(np.where(dates == pd.Timestamp(d))[0][0])
        ev = load_events()
        ev_days = pd.to_datetime(ev["trading_day"])
        past = ev_days[ev_days <= pd.Timestamp(d)]
        e_day = past.max()
        j = int(np.where(dates == e_day)[0][0])
        st.write(f"Last event day: **{e_day.date()}** ({', '.join(ev.loc[ev_days == e_day, 'event_id'])}); "
                 f"edge gate on that day = {g['gate'][j]:.3f}")
        c = st.columns(3)
        c[0].pyplot(draw_graph(g["A"][j - 1], f"day before event ({dates[j - 1].date()})"))
        c[1].pyplot(draw_graph(g["A"][j], f"event day ({e_day.date()})"))
        c[2].pyplot(draw_graph(g["A"][i], f"selected day ({d})"))
        st.write("Event-driven change on the event day (A − A without event edges):")
        st.dataframe(pd.DataFrame(g["A"][j] - g["A_noevent"][j], index=NODES, columns=NODES).style.format("{:+.4f}"))

elif page == "Event explorer":
    st.title("Event explorer")
    ev = load_events()
    eid = st.selectbox("Event", ev["event_id"].tolist()[::-1])
    e = ev[ev["event_id"] == eid].iloc[0]
    st.write(f"**{e.type}** at {e.datetime_ny} ET → trading day {e.trading_day}")
    st.caption(str(e.text)[:600])
    f = paths.RAW_INTRADAY / f"{eid}_gold.csv"
    if f.exists() and f.stat().st_size > 0:
        p = pd.read_csv(f)
        if len(p):
            t = pd.to_datetime(p["timestamp"], unit="ms", utc=True).dt.tz_convert("America/New_York")
            et = pd.Timestamp(e.datetime_utc).tz_convert("America/New_York")
            m = (t >= et - pd.Timedelta(minutes=60)) & (t <= pd.Timestamp(f"{e.trading_day} 17:00").tz_localize("America/New_York"))
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(t[m], p["close"][m], lw=0.8, color="goldenrod")
            for mins, lab in [(0, "event"), (5, "+5m"), (60, "+1h")]:
                ax.axvline(et + pd.Timedelta(minutes=mins), ls="--", lw=0.7, color="k")
                ax.text(et + pd.Timedelta(minutes=mins), ax.get_ylim()[1], lab, fontsize=7)
            ax.set_title("XAU/USD 1-minute (bid), New York time", fontsize=8)
            st.pyplot(fig)
    else:
        st.info("No intraday window for this event.")
    for mode in ["intraday", "daily"]:
        fr = paths.PROCESSED / f"event_signatures_{mode}_raw.csv"
        if fr.exists():
            sig = pd.read_csv(fr).set_index("event_id")
            if eid in sig.index:
                exp = json.loads((paths.PROCESSED / f"expected_reactions_{mode}.json").read_text())[e.type]
                st.subheader(f"Signature ({mode}, unscaled)")
                st.write(f"Expected gold mag_close for {e.type} (train mean): {exp['mag_close']:+.3f} | "
                         f"actual: {sig.loc[eid, 'mag_close']:+.3f}")
                st.dataframe(sig.loc[[eid]].T.rename(columns={eid: "value"}).style.format("{:.3f}"))
                break

else:
    st.title("Results")
    for f in ["SUMMARY.md", "ablation_table.md", "significance.md", "event_analysis.md", "backtest.md"]:
        p = paths.RESULTS / f
        if p.exists():
            with st.expander(f, expanded=(f == "SUMMARY.md")):
                st.markdown(p.read_text())
    for img in ["calibration.png", "graph_change_by_type.png", "permutation_importance.png"]:
        p = paths.FIGURES / img
        if p.exists():
            st.image(str(p), caption=img)
