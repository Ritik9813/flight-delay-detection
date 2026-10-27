"""Validation, cleaning, feature engineering and the shared sklearn pipeline.

The same ``build_pipeline`` object is used for training and inference, so
preprocessing can never drift between the two.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES, DELAY_THRESHOLD_MIN, INPUT_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS, NUMERIC_FEATURES, REQUIRED_INPUT_COLUMNS,
    TRAFFIC_LEVELS, WEATHER_CONDITIONS,
)

NUMERIC_RANGES = {
    "Distance_km": (1, 20000),
    "Temperature_C": (-60, 60),
    "Wind_Speed_kmh": (0, 300),
    "Visibility_km": (0, 100),
    "Previous_Delay_Min": (-60, 1440),
}
# Columns that may be blank in the TRAINING file (they are imputed later).
TRAINING_IMPUTABLE_COLUMNS = (
    "Temperature_C", "Wind_Speed_kmh", "Visibility_km",
    "Weather_Condition", "Traffic_Level",
)


class MissingColumnsError(ValueError):
    """Raised when required columns are absent from the input."""

    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)
        super().__init__("Missing required column(s): " + ", ".join(self.columns))


class InputValidationError(ValueError):
    """Raised when a single record fails validation. ``messages`` lists the problems."""

    def __init__(self, messages: list[str]):
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def time_to_minutes(series: pd.Series) -> pd.Series:
    """Convert 'HH:MM' / 'HH:MM:SS' strings (or time objects) to minutes after midnight."""
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
    td = pd.to_timedelta(s, errors="coerce")
    return td.dt.total_seconds() / 60.0


def _format_minutes(m: float):
    if pd.isna(m) or m < 0 or m >= 1440:
        return None
    return f"{int(m // 60):02d}:{int(m % 60):02d}"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_input_frame(
    df: pd.DataFrame, allow_missing: Iterable[str] = ()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate prediction inputs row by row.

    Returns ``(valid_rows, errors)``. ``valid_rows`` keeps the original row
    positions as its index and contains normalised values. ``errors`` has one
    line per rejected row with human-readable issues.

    Raises ``MissingColumnsError`` if a required column does not exist.
    ``allow_missing`` lists columns whose blanks are tolerated (training only).
    """
    missing_cols = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise MissingColumnsError(missing_cols)

    work = df.copy().reset_index(drop=True)
    n = len(work)
    reasons: list[list[str]] = [[] for _ in range(n)]
    allow = set(allow_missing)

    def flag(mask, message: str) -> None:
        arr = pd.Series(mask).fillna(False).to_numpy(dtype=bool)
        for i in np.flatnonzero(arr):
            reasons[i].append(message)

    # Free-text identifiers -------------------------------------------------
    for col in ("Airline", "Origin", "Destination"):
        s = work[col].astype("string").str.strip()
        blank = (s.isna() | (s == "")).fillna(True)
        flag(blank, f"{col} is missing")
        if col != "Airline":
            s = s.str.upper()
        work[col] = s.astype(object).where(~blank, np.nan)

    # Controlled vocabularies -----------------------------------------------
    for col, allowed in (("Weather_Condition", WEATHER_CONDITIONS), ("Traffic_Level", TRAFFIC_LEVELS)):
        s = work[col].astype("string").str.strip().str.title()
        miss = (s.isna() | (s == "")).fillna(True)
        invalid = (~miss) & (~s.isin(allowed)).fillna(True)
        if col not in allow:
            flag(miss, f"{col} is missing")
        flag(invalid, f"{col} must be one of {allowed}")
        work[col] = s.astype(object).where(~(miss | invalid), np.nan)

    # Numeric fields ----------------------------------------------------------
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in work.columns:
            work[col] = np.nan
        raw = work[col].replace(r"^\s*$", np.nan, regex=True)
        num = pd.to_numeric(raw, errors="coerce")
        missing = raw.isna()
        unparsable = (~missing) & num.isna()
        if col != "Previous_Delay_Min" and col not in allow:
            flag(missing, f"{col} is missing")
        flag(unparsable, f"{col} is not a number")
        flag(num.notna() & ((num < lo) | (num > hi)), f"{col} must be between {lo} and {hi}")
        work[col] = num.astype(float)

    # Date and times ----------------------------------------------------------
    date = pd.to_datetime(
        work["Departure_Date"].astype(str).str.strip(), format="ISO8601", errors="coerce"
    )
    flag(date.isna(), "Departure_Date is missing or not a valid date (use YYYY-MM-DD)")
    work["Departure_Date"] = date.dt.strftime("%Y-%m-%d")

    for col in ("Scheduled_Departure", "Scheduled_Arrival"):
        mins = time_to_minutes(work[col])
        formatted = [_format_minutes(m) for m in mins]
        flag([f is None for f in formatted],
             f"{col} is missing or not a valid time (use HH:MM, 00:00-23:59)")
        work[col] = pd.Series(formatted, index=work.index, dtype=object)

    # Cross-field and identity checks -----------------------------------------
    flag(work["Origin"].notna() & (work["Origin"] == work["Destination"]),
         "Origin and Destination must be different airports")
    if "Flight_ID" in work.columns:
        flag(work["Flight_ID"].notna() & work["Flight_ID"].duplicated(keep="first"),
             "Duplicate Flight_ID")

    bad = [i for i, r in enumerate(reasons) if r]
    flight_ids = work["Flight_ID"] if "Flight_ID" in work.columns else pd.Series([None] * n)
    errors = pd.DataFrame({
        "Row": [i + 1 for i in bad],
        "Flight_ID": [flight_ids.iloc[i] for i in bad],
        "Issues": ["; ".join(reasons[i]) for i in bad],
    })
    valid = work.drop(index=bad)
    return valid, errors


# --------------------------------------------------------------------------- #
# Feature engineering and pipeline
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn validated flight inputs into model features (no outcome columns used)."""
    out = pd.DataFrame(index=df.index)
    dep = time_to_minutes(df["Scheduled_Departure"])
    arr = time_to_minutes(df["Scheduled_Arrival"])
    date = pd.to_datetime(df["Departure_Date"].astype(str), format="ISO8601", errors="coerce")

    for col in ("Distance_km", "Temperature_C", "Wind_Speed_kmh", "Visibility_km"):
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    prev = (pd.to_numeric(df["Previous_Delay_Min"], errors="coerce").astype(float)
            if "Previous_Delay_Min" in df.columns else pd.Series(np.nan, index=df.index))
    out["Previous_Delay_Min"] = prev
    out["Previous_Delay_Known"] = prev.notna().astype(float)
    out["Dep_Hour"] = np.floor(dep / 60.0)
    out["Day_Of_Week"] = date.dt.dayofweek.astype(float)
    out["Month"] = date.dt.month.astype(float)
    out["Is_Weekend"] = (date.dt.dayofweek >= 5).astype(float)
    hour = out["Dep_Hour"]
    out["Is_Peak_Hour"] = (hour.between(6, 9) | hour.between(16, 20)).astype(float)
    out["Scheduled_Duration_Min"] = (arr - dep) % 1440  # handles overnight arrivals

    for col in CATEGORICAL_FEATURES:
        s = df[col]
        out[col] = s.astype(object).where(s.notna(), np.nan)
    return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(estimator) -> Pipeline:
    """Feature engineering + preprocessing + estimator as ONE fitted object."""
    return Pipeline([
        ("features", FunctionTransformer(engineer_features, validate=False)),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


def assert_no_leakage(columns: Iterable[str]) -> None:
    leaked = sorted(set(columns) & set(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(f"Target leakage: outcome columns used as features: {leaked}")


# --------------------------------------------------------------------------- #
# Training-data loading and cleaning
# --------------------------------------------------------------------------- #
def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it with: python generate_dataset.py"
        )
    return pd.read_csv(path)


def clean_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deduplicate, validate and label the training data. Returns (clean_df, report)."""
    needed = ["Flight_ID", "Departure_Delay_Min"]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise MissingColumnsError(absent)

    report = {"rows_in": int(len(df))}
    df = df.copy()
    dup_mask = df["Flight_ID"].duplicated(keep="first")
    report["duplicate_ids_removed"] = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)

    df["Departure_Delay_Min"] = pd.to_numeric(df["Departure_Delay_Min"], errors="coerce")
    no_target = df["Departure_Delay_Min"].isna()
    report["rows_missing_target"] = int(no_target.sum())
    df = df[~no_target].reset_index(drop=True)

    valid, errors = validate_input_frame(df, allow_missing=TRAINING_IMPUTABLE_COLUMNS)
    report["rows_invalid_dropped"] = int(len(errors))
    clean = df.loc[valid.index, ["Flight_ID", "Departure_Delay_Min"]].copy()
    clean = pd.concat([clean.reset_index(drop=True), valid.drop(columns=["Flight_ID", "Departure_Delay_Min"], errors="ignore").reset_index(drop=True)], axis=1)

    # The label is always derived from the delay minutes, never trusted blindly.
    derived = np.where(clean["Departure_Delay_Min"] >= DELAY_THRESHOLD_MIN, "Delayed", "On-Time")
    if "Delay_Status" in df.columns:
        original = df.loc[valid.index, "Delay_Status"].reset_index(drop=True)
        report["labels_inconsistent_with_minutes"] = int((original.astype(str) != derived).sum())
    clean["Delay_Status"] = derived
    report["rows_out"] = int(len(clean))
    report["missing_values_imputed"] = {
        c: int(clean[c].isna().sum()) for c in INPUT_FEATURE_COLUMNS if clean[c].isna().any()
    }
    return clean, report


def make_xy(clean: pd.DataFrame):
    """Split into features X, binary target y (1 = Delayed) and delay minutes."""
    X = clean[INPUT_FEATURE_COLUMNS].copy()
    assert_no_leakage(X.columns)
    y = (clean["Departure_Delay_Min"] >= DELAY_THRESHOLD_MIN).astype(int)
    return X, y, clean["Departure_Delay_Min"].astype(float)


def add_scheduled_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Dep_Hour"] = np.floor(time_to_minutes(out["Scheduled_Departure"]) / 60.0)
    return out


def dataset_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_flight_ids": int(df["Flight_ID"].duplicated().sum()) if "Flight_ID" in df else 0,
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
    }
