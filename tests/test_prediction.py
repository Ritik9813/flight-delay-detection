import json

import joblib
import pandas as pd
import pytest

from generate_dataset import generate_dataset
from src.history import append_history, build_history_records, clear_history, load_history
from src.preprocessing import InputValidationError, MissingColumnsError
from src.prediction import (
    ModelLoadError, ModelNotFoundError, load_model_bundle, predict_dataframe,
    predict_single, risk_category, validate_risk_thresholds,
)
from train_model import train_and_save


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    folder = tmp_path_factory.mktemp("artifacts")
    data_path = folder / "flights.csv"
    generate_dataset(500, seed=11).to_csv(data_path, index=False)
    model_path, metrics_path = folder / "model.joblib", folder / "metrics.json"
    metrics = train_and_save(data_path, model_path, metrics_path, verbose=False)
    return {"model": model_path, "metrics_path": metrics_path, "metrics": metrics}


@pytest.fixture(scope="module")
def bundle(artifacts):
    return load_model_bundle(artifacts["model"])


RECORD = dict(
    Airline="SkyJet Airways", Origin="JFK", Destination="ORD", Departure_Date="2026-03-02",
    Scheduled_Departure="18:30", Scheduled_Arrival="20:10", Distance_km=1190,
    Weather_Condition="Thunderstorm", Temperature_C=22.0, Wind_Speed_kmh=40.0,
    Visibility_km=3.0, Traffic_Level="High", Previous_Delay_Min=45,
)
CALM = dict(RECORD, Weather_Condition="Clear", Wind_Speed_kmh=6.0, Visibility_km=10.0,
            Traffic_Level="Low", Previous_Delay_Min=0, Scheduled_Departure="06:30", Scheduled_Arrival="08:10")


# ----------------------------- model loading ---------------------------------
def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError) as exc:
        load_model_bundle(tmp_path / "nope.joblib")
    assert "python train_model.py" in str(exc.value)


def test_corrupt_or_invalid_model_file_is_rejected(tmp_path):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a model")
    with pytest.raises(ModelLoadError):
        load_model_bundle(corrupt)
    wrong = tmp_path / "wrong.joblib"
    joblib.dump({"hello": "world"}, wrong)
    with pytest.raises(ModelLoadError):
        load_model_bundle(wrong)


def test_bundle_loads_with_expected_contents(bundle):
    assert {"classifier", "regressor", "model_name", "baselines", "categories"} <= set(bundle)
    assert bundle["model_name"] in {"Logistic Regression", "Random Forest", "Gradient Boosting"}


# ----------------------------- training outputs ------------------------------
def test_metrics_file_is_complete_and_computed(artifacts):
    on_disk = json.loads(artifacts["metrics_path"].read_text())
    assert on_disk["split"]["train_rows"] + on_disk["split"]["test_rows"] == 500
    assert set(on_disk["models"]) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    for result in on_disk["models"].values():
        test = result["test"]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= test[key] <= 1.0
        assert sum(map(sum, test["confusion_matrix"])) == on_disk["split"]["test_rows"]
        assert 0.0 <= test["roc_auc"] <= 1.0
    assert on_disk["regression"]["mae"] >= 0
    assert "Actual_Departure" in on_disk["features"]["excluded_for_leakage"]


def test_selected_model_has_best_cv_f1(artifacts):
    models = artifacts["metrics"]["models"]
    best = max(models, key=lambda n: models[n]["cv_f1_mean"])
    assert artifacts["metrics"]["selection"]["selected_model"] == best


# ----------------------------- single prediction -----------------------------
def test_single_prediction_output_is_well_formed(bundle):
    result = predict_single(bundle, RECORD)
    assert result["Predicted_Status"] in {"Delayed", "On-Time"}
    assert 0.0 <= result["Delay_Probability"] <= 1.0
    assert result["Risk_Category"] in {"Low", "Medium", "High"}
    assert result["Estimated_Delay_Min"] >= 0
    assert isinstance(result["Explanation"], list) and result["Explanation"]


def test_risky_flight_scores_higher_than_calm_flight(bundle):
    assert predict_single(bundle, RECORD)["Delay_Probability"] > predict_single(bundle, CALM)["Delay_Probability"]


def test_previous_delay_can_be_omitted(bundle):
    record = {k: v for k, v in CALM.items() if k != "Previous_Delay_Min"}
    assert 0.0 <= predict_single(bundle, record)["Delay_Probability"] <= 1.0


def test_invalid_single_input_raises_with_messages(bundle):
    with pytest.raises(InputValidationError) as exc:
        predict_single(bundle, dict(RECORD, Distance_km=-10, Traffic_Level="Huge"))
    text = " ".join(exc.value.messages)
    assert "Distance_km" in text and "Traffic_Level" in text


def test_missing_field_in_single_input_raises(bundle):
    record = dict(RECORD)
    del record["Airline"]
    with pytest.raises(InputValidationError):
        predict_single(bundle, record)


# ----------------------------- batch prediction ------------------------------
def test_batch_prediction_splits_valid_and_invalid_rows(bundle):
    df = pd.DataFrame([RECORD, dict(RECORD, Scheduled_Departure="99:99"), CALM])
    df["Flight_ID"] = ["A", "B", "C"]
    results, errors = predict_dataframe(bundle, df)
    assert results["Flight_ID"].tolist() == ["A", "C"]
    assert errors["Flight_ID"].tolist() == ["B"]
    assert {"Predicted_Status", "Delay_Probability", "Risk_Category", "Estimated_Delay_Min"} <= set(results.columns)


def test_batch_ignores_actual_outcome_columns(bundle):
    df = pd.DataFrame([RECORD])
    baseline, _ = predict_dataframe(bundle, df)
    df["Departure_Delay_Min"] = 500
    df["Delay_Status"] = "Delayed"
    df["Actual_Departure"] = "2026-03-02 23:59"
    with_leaks, _ = predict_dataframe(bundle, df)
    assert baseline["Delay_Probability"].iloc[0] == with_leaks["Delay_Probability"].iloc[0]


def test_batch_missing_required_columns_raises(bundle):
    with pytest.raises(MissingColumnsError):
        predict_dataframe(bundle, pd.DataFrame([RECORD]).drop(columns=["Distance_km"]))


def test_batch_with_only_invalid_rows_returns_empty_results(bundle):
    results, errors = predict_dataframe(bundle, pd.DataFrame([dict(RECORD, Traffic_Level="x")]))
    assert results.empty and len(errors) == 1


# ----------------------------- risk categories -------------------------------
@pytest.mark.parametrize("p, expected", [(0.0, "Low"), (0.29, "Low"), (0.30, "Medium"), (0.59, "Medium"), (0.60, "High"), (1.0, "High")])
def test_default_risk_bands(p, expected):
    assert risk_category(p) == expected


def test_risk_bands_are_configurable_and_validated():
    assert risk_category(0.5, {"low_max": 0.6, "high_min": 0.9}) == "Low"
    with pytest.raises(ValueError):
        validate_risk_thresholds({"low_max": 0.8, "high_min": 0.2})


# ----------------------------- history ---------------------------------------
def test_history_roundtrip_and_clear(bundle, tmp_path):
    path = tmp_path / "history.csv"
    assert load_history(path).empty
    results, _ = predict_dataframe(bundle, pd.DataFrame([RECORD, CALM]))
    assert append_history(build_history_records(results), path) == 2
    history = load_history(path)
    assert len(history) == 2 and "JFK->ORD" in history["Input_Summary"].iloc[0]
    clear_history(path)
    assert load_history(path).empty


def test_model_trained_with_different_sklearn_version_is_rejected(bundle, tmp_path):
    stale = dict(bundle, sklearn_version="0.0.1")
    path = tmp_path / "stale.joblib"
    joblib.dump(stale, path)
    with pytest.raises(ModelLoadError) as exc:
        load_model_bundle(path)
    assert "python train_model.py" in str(exc.value)
