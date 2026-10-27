"""Local prediction history (a plain CSV next to the dataset).

Only the minimum needed to review a prediction is stored: no names, contact
details or free-text fields. The file never leaves your machine.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import HISTORY_PATH

HISTORY_COLUMNS = [
    "Timestamp", "Flight_ID", "Input_Summary", "Predicted_Status",
    "Delay_Probability", "Risk_Category", "Estimated_Delay_Min",
]


def summarize_inputs(row: dict) -> str:
    return (
        f"{row.get('Airline')} {row.get('Origin')}->{row.get('Destination')} "
        f"{row.get('Departure_Date')} {row.get('Scheduled_Departure')} | "
        f"{row.get('Weather_Condition')}, traffic {row.get('Traffic_Level')}"
    )


def build_history_records(results: pd.DataFrame) -> list[dict]:
    """Convert prediction result rows into history records."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    records = []
    for row in results.to_dict("records"):
        flight_id = row.get("Flight_ID")
        records.append({
            "Timestamp": timestamp,
            "Flight_ID": "" if flight_id is None or pd.isna(flight_id) else flight_id,
            "Input_Summary": summarize_inputs(row),
            "Predicted_Status": row.get("Predicted_Status"),
            "Delay_Probability": row.get("Delay_Probability"),
            "Risk_Category": row.get("Risk_Category"),
            "Estimated_Delay_Min": row.get("Estimated_Delay_Min"),
        })
    return records


def append_history(records: list[dict], path=HISTORY_PATH) -> int:
    if not records:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(records, columns=HISTORY_COLUMNS)
    new.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(new)


def load_history(path=HISTORY_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return df.reindex(columns=HISTORY_COLUMNS).iloc[::-1].reset_index(drop=True)  # newest first


def clear_history(path=HISTORY_PATH) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
