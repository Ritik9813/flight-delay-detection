import numpy as np
import pandas as pd
import pytest

from generate_dataset import generate_dataset
from src.config import (
    DATASET_COLUMNS, DELAY_THRESHOLD_MIN, INPUT_FEATURE_COLUMNS, LEAKAGE_COLUMNS,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.preprocessing import (
    MissingColumnsError, assert_no_leakage, build_pipeline, clean_training_data,
    engineer_features, make_xy, time_to_minutes, validate_input_frame,
)


@pytest.fixture(scope="module")
def raw():
    return generate_dataset(500, seed=7)


def valid_record(**overrides):
    record = dict(
        Airline="SkyJet Airways", Origin="JFK", Destination="ORD", Departure_Date="2026-03-02",
        Scheduled_Departure="08:30", Scheduled_Arrival="10:15", Distance_km=1190,
        Weather_Condition="Clear", Temperature_C=12.0, Wind_Speed_kmh=10.0,
        Visibility_km=10.0, Traffic_Level="Low", Previous_Delay_Min=0,
    )
    record.update(overrides)
    return record


# ----------------------------- dataset ---------------------------------------
def test_dataset_has_exactly_500_unique_flights_and_expected_columns(raw):
    assert len(raw) == 500
    assert raw["Flight_ID"].is_unique
    assert list(raw.columns) == DATASET_COLUMNS


def test_label_is_derived_from_delay_minutes(raw):
    expected = np.where(raw["Departure_Delay_Min"] >= DELAY_THRESHOLD_MIN, "Delayed", "On-Time")
    assert (raw["Delay_Status"].to_numpy() == expected).all()
    assert raw["Delay_Status"].nunique() == 2


def test_dataset_generation_is_reproducible():
    pd.testing.assert_frame_equal(generate_dataset(60, seed=3), generate_dataset(60, seed=3))


# ----------------------------- cleaning --------------------------------------
def test_duplicate_flight_ids_are_removed(raw):
    duplicated = pd.concat([raw, raw.iloc[:10]], ignore_index=True)
    clean, report = clean_training_data(duplicated)
    assert report["duplicate_ids_removed"] == 10
    assert len(clean) == 500


def test_clean_training_data_requires_target_columns(raw):
    with pytest.raises(MissingColumnsError):
        clean_training_data(raw.drop(columns=["Departure_Delay_Min"]))


def test_missing_numeric_values_are_kept_for_imputation_in_training(raw):
    clean, report = clean_training_data(raw)
    assert len(clean) == 500
    assert report["missing_values_imputed"]  # generator injects a few blanks


def test_inconsistent_labels_are_recomputed(raw):
    tampered = raw.copy()
    tampered.loc[0, "Delay_Status"] = "On-Time" if tampered.loc[0, "Delay_Status"] == "Delayed" else "Delayed"
    clean, report = clean_training_data(tampered)
    assert report["labels_inconsistent_with_minutes"] == 1
    expected = "Delayed" if clean.loc[0, "Departure_Delay_Min"] >= DELAY_THRESHOLD_MIN else "On-Time"
    assert clean.loc[0, "Delay_Status"] == expected


# ----------------------------- validation ------------------------------------
def test_missing_required_columns_raise():
    df = pd.DataFrame([valid_record()]).drop(columns=["Airline", "Traffic_Level"])
    with pytest.raises(MissingColumnsError) as exc:
        validate_input_frame(df)
    assert set(exc.value.columns) == {"Airline", "Traffic_Level"}


def test_valid_row_passes_and_previous_delay_is_optional():
    record = valid_record()
    del record["Previous_Delay_Min"]
    valid, errors = validate_input_frame(pd.DataFrame([record]))
    assert len(valid) == 1 and errors.empty
    assert np.isnan(valid["Previous_Delay_Min"].iloc[0])


@pytest.mark.parametrize("overrides, fragment", [
    ({"Distance_km": -5}, "Distance_km"),
    ({"Distance_km": "far"}, "not a number"),
    ({"Temperature_C": 500}, "Temperature_C"),
    ({"Wind_Speed_kmh": -1}, "Wind_Speed_kmh"),
    ({"Visibility_km": None}, "Visibility_km is missing"),
    ({"Traffic_Level": "Huge"}, "Traffic_Level"),
    ({"Weather_Condition": "Tornado"}, "Weather_Condition"),
    ({"Scheduled_Departure": "25:61"}, "Scheduled_Departure"),
    ({"Scheduled_Arrival": "soon"}, "Scheduled_Arrival"),
    ({"Departure_Date": "not-a-date"}, "Departure_Date"),
    ({"Destination": "JFK"}, "different airports"),
    ({"Airline": ""}, "Airline is missing"),
    ({"Previous_Delay_Min": 99999}, "Previous_Delay_Min"),
])
def test_invalid_values_are_rejected_with_helpful_message(overrides, fragment):
    valid, errors = validate_input_frame(pd.DataFrame([valid_record(**overrides)]))
    assert valid.empty
    assert fragment in errors["Issues"].iloc[0]


def test_validation_keeps_good_rows_and_reports_bad_ones():
    df = pd.DataFrame([valid_record(), valid_record(Traffic_Level="??"), valid_record()])
    valid, errors = validate_input_frame(df)
    assert len(valid) == 2
    assert errors["Row"].tolist() == [2]


def test_text_inputs_are_normalised():
    valid, _ = validate_input_frame(pd.DataFrame([valid_record(Origin=" jfk ", Weather_Condition="rain", Traffic_Level="HIGH")]))
    row = valid.iloc[0]
    assert (row["Origin"], row["Weather_Condition"], row["Traffic_Level"]) == ("JFK", "Rain", "High")


def test_duplicate_flight_ids_in_batch_are_flagged():
    df = pd.DataFrame([valid_record(Flight_ID="A1"), valid_record(Flight_ID="A1")])
    valid, errors = validate_input_frame(df)
    assert len(valid) == 1 and "Duplicate Flight_ID" in errors["Issues"].iloc[0]


# ----------------------------- features / leakage ----------------------------
def test_time_parsing_handles_hhmm_and_seconds():
    minutes = time_to_minutes(pd.Series(["08:30", "8:05", "23:59:00", "bad", None]))
    assert minutes.iloc[:3].tolist() == [510.0, 485.0, 1439.0]
    assert minutes.iloc[3:].isna().all()


def test_engineered_features_have_expected_columns_and_values():
    valid, _ = validate_input_frame(pd.DataFrame([valid_record(Departure_Date="2026-03-07", Scheduled_Departure="23:30", Scheduled_Arrival="01:10")]))
    f = engineer_features(valid).iloc[0]
    assert list(engineer_features(valid).columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert f["Dep_Hour"] == 23 and f["Month"] == 3
    assert f["Day_Of_Week"] == 5 and f["Is_Weekend"] == 1  # Saturday
    assert f["Scheduled_Duration_Min"] == 100  # overnight arrival handled


def test_model_inputs_exclude_outcome_columns(raw):
    clean, _ = clean_training_data(raw)
    X, y, _ = make_xy(clean)
    assert set(X.columns) == set(INPUT_FEATURE_COLUMNS)
    assert not set(X.columns) & set(LEAKAGE_COLUMNS)
    assert set(y.unique()) <= {0, 1}
    with pytest.raises(ValueError):
        assert_no_leakage(["Airline", "Actual_Departure"])


def test_pipeline_runs_with_unseen_category_and_missing_previous_delay(raw):
    from sklearn.linear_model import LogisticRegression
    clean, _ = clean_training_data(raw)
    X, y, _ = make_xy(clean)
    pipeline = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    new = pd.DataFrame([valid_record(Airline="Brand New Air", Previous_Delay_Min=None)])
    valid, _ = validate_input_frame(new)
    proba = pipeline.predict_proba(valid[INPUT_FEATURE_COLUMNS])[0, 1]
    assert 0.0 <= proba <= 1.0
