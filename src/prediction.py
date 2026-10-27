"""Model loading, single/batch prediction, risk bands and explanations."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import (
    DEFAULT_RISK_THRESHOLDS, INPUT_FEATURE_COLUMNS, MODEL_PATH,
)
from src.preprocessing import (
    InputValidationError, MissingColumnsError, time_to_minutes, validate_input_frame,
)

REQUIRED_BUNDLE_KEYS = ("classifier", "model_name", "decision_threshold", "baselines", "categories")


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model artifact is missing."""


class ModelLoadError(RuntimeError):
    """Raised when the model artifact exists but cannot be used."""


def load_model_bundle(path=MODEL_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at '{path}'. Train it first with: python train_model.py"
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # corrupt file, version mismatch, missing module...
        raise ModelLoadError(
            f"Could not load '{path}' ({exc}). Re-train the model with: python train_model.py"
        ) from exc
    if not isinstance(bundle, dict) or any(k not in bundle for k in REQUIRED_BUNDLE_KEYS):
        raise ModelLoadError(
            f"'{path}' is not a valid model bundle. Re-train the model with: python train_model.py"
        )
    saved = str(bundle.get("sklearn_version", "unknown"))
    if saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        raise ModelLoadError(
            f"The saved model was trained with scikit-learn {saved}, but scikit-learn "
            f"{sklearn.__version__} is installed, so it may not work correctly. "
            "Re-train it on this machine with: python train_model.py"
        )
    return bundle


def validate_risk_thresholds(thresholds: dict | None) -> dict:
    t = dict(DEFAULT_RISK_THRESHOLDS if thresholds is None else thresholds)
    low, high = t.get("low_max"), t.get("high_min")
    if low is None or high is None or not (0.0 <= low < high <= 1.0):
        raise ValueError("Risk thresholds must satisfy 0 <= low_max < high_min <= 1.")
    return {"low_max": float(low), "high_min": float(high)}


def risk_category(probability: float, thresholds: dict | None = None) -> str:
    """Low: p < low_max; High: p >= high_min; otherwise Medium."""
    t = validate_risk_thresholds(thresholds)
    if probability < t["low_max"]:
        return "Low"
    if probability >= t["high_min"]:
        return "High"
    return "Medium"


def _delay_probability(pipeline, X: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def _estimated_minutes(bundle: dict, X: pd.DataFrame):
    regressor = bundle.get("regressor")
    if regressor is None:
        return [None] * len(X)
    return np.round(np.clip(regressor.predict(X), 0, None), 1)


def predict_dataframe(bundle: dict, df: pd.DataFrame, risk_thresholds: dict | None = None):
    """Validate and predict many flights.

    Returns ``(results, errors)``. Rows that fail validation are excluded from
    ``results`` and described in ``errors``. Raises ``MissingColumnsError`` if
    required columns are absent. Actual-delay/actual-departure columns are
    never read.
    """
    thresholds = validate_risk_thresholds(risk_thresholds)
    valid, errors = validate_input_frame(df)
    base_cols = (["Flight_ID"] if "Flight_ID" in valid.columns else []) + INPUT_FEATURE_COLUMNS
    if valid.empty:
        empty = pd.DataFrame(columns=["Row"] + base_cols + [
            "Predicted_Status", "Delay_Probability", "Risk_Category", "Estimated_Delay_Min"])
        return empty, errors

    X = valid[INPUT_FEATURE_COLUMNS]
    probability = _delay_probability(bundle["classifier"], X)
    status = np.where(probability >= bundle["decision_threshold"], "Delayed", "On-Time")
    results = valid[base_cols].copy()
    results.insert(0, "Row", valid.index + 1)
    results["Predicted_Status"] = status
    results["Delay_Probability"] = np.round(probability, 4)
    results["Risk_Category"] = [risk_category(p, thresholds) for p in probability]
    results["Estimated_Delay_Min"] = _estimated_minutes(bundle, X)
    return results.reset_index(drop=True), errors


# --------------------------------------------------------------------------- #
# Explanation (occlusion against a "typical flight")
# --------------------------------------------------------------------------- #
def explain_prediction(bundle: dict, row: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Estimate how much each key input moves the delay probability.

    For each factor, the input is replaced by its typical (median/mode) value
    from the training data and the change is recorded both in probability
    points and as an odds multiplier (log-odds are used for ranking because raw
    probabilities saturate near 0% / 100%). This is a local sensitivity
    estimate, not proof of causation.
    """
    base = bundle["baselines"]
    row = row[INPUT_FEATURE_COLUMNS].iloc[[0]].copy()
    p_actual = float(_delay_probability(bundle["classifier"], row)[0])

    variants, meta = [], []

    def add(name: str, value_label, baseline_label, mutate):
        counterfactual = row.copy()
        mutate(counterfactual)
        variants.append(counterfactual)
        meta.append((name, value_label, baseline_label))

    r = row.iloc[0]
    for col, label in (("Weather_Condition", "Weather"), ("Traffic_Level", "Traffic level"), ("Airline", "Airline")):
        if r[col] != base[col]:
            add(label, r[col], base[col], lambda d, c=col: d.__setitem__(c, base[c]))
    for col, label, unit in (
        ("Wind_Speed_kmh", "Wind speed", "km/h"), ("Visibility_km", "Visibility", "km"),
        ("Temperature_C", "Temperature", "°C"), ("Distance_km", "Distance", "km"),
    ):
        if pd.notna(r[col]) and abs(float(r[col]) - base[col]) > 1e-9:
            add(label, f"{float(r[col]):g} {unit}", f"{base[col]:.4g} {unit}",
                lambda d, c=col: d.__setitem__(c, base[c]))
    if pd.notna(r["Previous_Delay_Min"]) and abs(float(r["Previous_Delay_Min"]) - base["Previous_Delay_Min"]) > 1e-9:
        add("Previous delay", f"{float(r['Previous_Delay_Min']):g} min", f"{base['Previous_Delay_Min']:.4g} min",
            lambda d: d.__setitem__("Previous_Delay_Min", base["Previous_Delay_Min"]))

    dep = float(time_to_minutes(pd.Series([r["Scheduled_Departure"]])).iloc[0])
    base_dep = float(time_to_minutes(pd.Series([base["Scheduled_Departure"]])).iloc[0])
    if abs(dep - base_dep) > 1e-9:
        arr = float(time_to_minutes(pd.Series([r["Scheduled_Arrival"]])).iloc[0])

        def shift_departure(d, dep=dep, arr=arr, base_dep=base_dep):
            new_arr = (arr + (base_dep - dep)) % 1440
            d["Scheduled_Departure"] = f"{int(base_dep // 60):02d}:{int(base_dep % 60):02d}"
            d["Scheduled_Arrival"] = f"{int(new_arr // 60):02d}:{int(new_arr % 60):02d}"

        add("Departure time", r["Scheduled_Departure"], base["Scheduled_Departure"], shift_departure)

    if not variants:
        return []
    p_counter = _delay_probability(bundle["classifier"], pd.concat(variants, ignore_index=True))

    def logit(p: float) -> float:
        p = min(max(float(p), 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    factors = []
    for (name, value, typical), p_cf in zip(meta, p_counter):
        shift = logit(p_actual) - logit(p_cf)  # change in log-odds vs. a typical flight
        factors.append({
            "factor": name,
            "value": str(value),
            "typical_value": str(typical),
            "impact_points": round((p_actual - float(p_cf)) * 100.0, 1),
            "odds_multiplier": round(float(np.exp(shift)), 2),
            "log_odds_shift": round(shift, 3),
            "direction": "raises" if shift > 0 else "lowers",
        })
    factors.sort(key=lambda f: abs(f["log_odds_shift"]), reverse=True)
    return factors[:top_n]


def predict_single(bundle: dict, record: dict, risk_thresholds: dict | None = None) -> dict:
    """Predict one flight. Raises ``InputValidationError`` with readable messages."""
    absent = [c for c in INPUT_FEATURE_COLUMNS if c != "Previous_Delay_Min" and c not in record]
    if absent:
        raise InputValidationError([f"Missing field: {c}" for c in absent])
    frame = pd.DataFrame([record])
    try:
        results, errors = predict_dataframe(bundle, frame, risk_thresholds)
    except MissingColumnsError as exc:
        raise InputValidationError([str(exc)]) from exc
    if not errors.empty:
        raise InputValidationError(errors["Issues"].iloc[0].split("; "))

    out = results.iloc[0].to_dict()
    valid, _ = validate_input_frame(frame)
    out["Explanation"] = explain_prediction(bundle, valid)
    return out
