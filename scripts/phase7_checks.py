"""Phase 7 checks: intraday coverage per type and year, and 1-minute gold paths for 3 events."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.download_intraday import event_window_utc
from src.features.intraday_signature import load_window
from src.utils import paths

q = pd.read_csv(paths.PROCESSED / "intraday_quality.csv")
print("valid intraday gold window (% of events):")
print((q.pivot_table(index="type", columns="year", values="ok_gold", aggfunc="mean") * 100).round(0).to_string())
print("overall per type:", (q.groupby("type")["ok_gold"].mean() * 100).round(1).to_dict(),
      f"| all {q['ok_gold'].mean():.1%}")
print("cross-asset ok rate:", q[["ok_usd", "ok_equities", "ok_oil"]].mean().round(3).to_dict(),
      "| usd source:", q["usd_source"].value_counts().to_dict())

ev = pd.read_csv(paths.EVENTS_CSV, parse_dates=["datetime_utc"]).set_index("event_id")
sig = pd.read_csv(paths.PROCESSED / "event_signatures_intraday_raw.csv").set_index("event_id")
pick = ["FOMC_20220615", "CPI_20221110", "NFP_20200508"]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, eid in zip(axes, pick):
    e = ev.loc[eid]
    start, end = event_window_utc(e, 60)
    s = load_window(eid, "gold", start, end)
    t = s.index.tz_convert("America/New_York")
    et = pd.Timestamp(e["datetime_utc"]).tz_convert("America/New_York")
    ax.plot(t, s.values, lw=0.7, color="goldenrod")
    for m, lab in [(0, "event"), (5, "+5m"), (60, "+1h")]:
        ax.axvline(et + pd.Timedelta(minutes=m), ls="--", lw=0.7, color="k")
        ax.text(et + pd.Timedelta(minutes=m), s.max(), lab, fontsize=7)
    r = sig.loc[eid]
    ax.set_title(f"{eid}: latency {r.latency_min:.0f}m, mag5m {r.mag_5m:+.1f}, mag1h {r.mag_1h:+.1f}, "
                 f"close {r.mag_close:+.1f}, pers {r.persistence:+.2f}", fontsize=7)
    ax.tick_params(labelsize=6)
plt.tight_layout(); plt.savefig(paths.FIGURES / "phase7_event_paths.png", dpi=110)
cols = ["dir_5m", "dir_1h", "dir_close", "xa_usd_dir", "xa_usd_mag", "has_intraday"]
print(sig.loc[pick + ["FOMC_20130619", "FOMC_20200315"], cols].round(2).to_string())
