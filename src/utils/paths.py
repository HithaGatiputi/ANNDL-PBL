"""All file-system paths used by the project, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_DAILY = RAW / "daily"
RAW_INTRADAY = RAW / "intraday"
RAW_EVENTS = RAW / "events"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS = REPORTS / "results"
LOGS = ROOT / "logs"

DAILY_PARQUET = INTERIM / "daily.parquet"
EVENTS_CSV = INTERIM / "events.csv"
REGIMES_PARQUET = INTERIM / "regimes.parquet"
GOLD_CLOSE_PARQUET = INTERIM / "gold_close_1700.parquet"
SIG_DAILY = PROCESSED / "signatures_daily.parquet"
SIG_INTRADAY = PROCESSED / "signatures_intraday.parquet"
EXPECTED_REACTIONS = PROCESSED / "expected_reactions.json"
EVENT_EMB = PROCESSED / "event_emb.parquet"
EVENT_EMB_DAILY = PROCESSED / "event_emb_daily.parquet"


def ensure_dirs() -> None:
    """Create every output directory if missing."""
    for p in [RAW_DAILY, RAW_INTRADAY, RAW_EVENTS, INTERIM, PROCESSED, FIGURES, RESULTS, LOGS]:
        p.mkdir(parents=True, exist_ok=True)
