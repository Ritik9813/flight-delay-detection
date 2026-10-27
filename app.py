"""Flight Delay Detection System - Streamlit dashboard.

Run with:  streamlit run app.py
The app only LOADS the trained model; it never retrains it.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st

from src.config import (
    DATA_PATH, DEFAULT_RISK_THRESHOLDS, DELAY_THRESHOLD_MIN, HISTORY_PATH,
    INPUT_FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH, REQUIRED_INPUT_COLUMNS,
    TRAFFIC_LEVELS, WEATHER_CONDITIONS,
)
from src.history import append_history, build_history_records, clear_history, load_history
from src.prediction import (
    ModelLoadError, ModelNotFoundError, load_model_bundle, predict_dataframe, predict_single,
)
from src.preprocessing import InputValidationError, MissingColumnsError, dataset_quality_report
from src import visualization as viz

st.set_page_config(page_title="Flight Delay Detection System", page_icon="✈️", layout="wide")

MAX_BATCH_ROWS = 50_000
SYNTHETIC_NOTE = (
    "The bundled dataset (flights_500.csv) it is kaggle-sourced synthetic data, generated to resemble real flight patterns. "
    "airlines, not actual airline data."
)

PAGES = [
    "Home Dashboard", "Dataset Explorer", "Flight Delay Prediction", "Batch Prediction",
    "Analytics", "Model Performance", "Prediction History",
]

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(120deg, #0B2545 0%, #1F78C8 100%); color: white;
           padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;}
    .hero h1 {color: white; margin: 0; font-size: 2.1rem;}
    .hero p {color: #D6E6F7; margin: .3rem 0 0 0;}
    .badge {display:inline-block; padding:.35rem .9rem; border-radius:999px; color:white;
            font-weight:700; font-size:1.05rem;}
    .badge-delayed {background:#E4572E;} .badge-ontime {background:#2A9D8F;}
    div[data-testid="stMetric"] {background:#F4F8FC; border:1px solid #E3EAF3;
           padding:.8rem 1rem; border-radius:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Compatibility helpers (Streamlit renamed use_container_width -> width)
# --------------------------------------------------------------------------- #
def show_table(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def show_chart(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>✈️ {title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _read_dataset(mtime: float) -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def get_dataset() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        return None
    df = _read_dataset(DATA_PATH.stat().st_mtime)
    return df.copy()


@st.cache_resource(show_spinner=False)
def _load_bundle(mtime: float) -> dict:
    return load_model_bundle(MODEL_PATH)


def get_bundle() -> tuple[dict | None, str | None]:
    try:
        if not MODEL_PATH.exists():
            raise ModelNotFoundError(f"Trained model not found at '{MODEL_PATH.name}'.")
        return _load_bundle(MODEL_PATH.stat().st_mtime), None
    except (ModelNotFoundError, ModelLoadError) as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def _read_metrics(mtime: float) -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def get_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return _read_metrics(METRICS_PATH.stat().st_mtime)
    except (json.JSONDecodeError, OSError):
        return None


def require_dataset() -> pd.DataFrame | None:
    df = get_dataset()
    if df is None:
        st.error("Dataset not found.")
        st.code("python generate_dataset.py", language="bash")
    return df


def require_model() -> dict | None:
    bundle, error = get_bundle()
    if bundle is None:
        st.error(f"{error}")
        st.markdown("Create the model artifacts, then reload this page:")
        st.code("python generate_dataset.py\npython train_model.py", language="bash")
    return bundle


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    hero("Flight Delay Detection System",
         "Machine-learning delay prediction from historical flight, weather and traffic data.")
    st.info("This is a **prediction model**, not a live flight tracker. It does not know the current "
            "status of any real flight. " + SYNTHETIC_NOTE)

    df = require_dataset()
    if df is None:
        return
    total = len(df)
    delayed = int((df["Delay_Status"] == "Delayed").sum())
    metrics = get_metrics()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total flights", f"{total:,}")
    c2.metric("Delayed flights", f"{delayed:,}")
    c3.metric("On-time flights", f"{total - delayed:,}")
    c4.metric("Delay percentage", f"{delayed / total:.1%}" if total else "n/a")
    if metrics:
        selected = metrics["selection"]["selected_model"]
        f1 = metrics["models"][selected]["test"]["f1"]
        c5.metric("Selected model", selected, f"Test F1 {f1:.2f}", delta_color="off")
    else:
        c5.metric("Selected model", "Not trained")

    if not metrics:
        st.warning("No trained model found. Run `python generate_dataset.py` then `python train_model.py`.")
    left, right = st.columns([1, 1])
    with left:
        show_chart(viz.status_chart(df))
    with right:
        st.subheader("How it works")
        st.markdown(
            f"""
1. Flights with an actual departure delay of **{DELAY_THRESHOLD_MIN}+ minutes** are labelled *Delayed*.
2. Three classifiers are compared by cross-validated F1; the best is selected.
3. Only information known **before departure** is used as input (no actual times or delay minutes).
4. Use **Flight Delay Prediction** for one flight or **Batch Prediction** for a CSV.

Probabilities are model estimates, not guarantees.
"""
        )


def page_explorer() -> None:
    hero("Dataset Explorer", "Search, filter and inspect the flight records.")
    df = require_dataset()
    if df is None:
        return
    st.caption(SYNTHETIC_NOTE)

    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        airlines = c1.multiselect("Airline", sorted(df["Airline"].dropna().unique()))
        origins = c2.multiselect("Origin", sorted(df["Origin"].dropna().unique()))
        dests = c3.multiselect("Destination", sorted(df["Destination"].dropna().unique()))
        statuses = c4.multiselect("Delay status", ["On-Time", "Delayed"])
        search = st.text_input("Search Flight_ID", placeholder="e.g. FL0042")

    view = df
    if airlines:
        view = view[view["Airline"].isin(airlines)]
    if origins:
        view = view[view["Origin"].isin(origins)]
    if dests:
        view = view[view["Destination"].isin(dests)]
    if statuses:
        view = view[view["Delay_Status"].isin(statuses)]
    if search.strip():
        view = view[view["Flight_ID"].astype(str).str.contains(search.strip(), case=False, regex=False)]

    st.write(f"Showing **{len(view):,}** of {len(df):,} flights")
    show_table(view)
    st.download_button("Download filtered dataset (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="flights_filtered.csv", mime="text/csv")

    st.subheader("Data quality")
    quality = dataset_quality_report(df)
    q1, q2, q3 = st.columns(3)
    q1.metric("Duplicate rows", quality["duplicate_rows"])
    q2.metric("Duplicate Flight_IDs", quality["duplicate_flight_ids"])
    q3.metric("Columns with missing values", sum(v > 0 for v in quality["missing_by_column"].values()))
    missing = pd.DataFrame({"Column": list(quality["missing_by_column"]),
                            "Missing values": list(quality["missing_by_column"].values())})
    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values per column**")
        show_table(missing)
    with right:
        st.markdown("**Summary statistics (numeric)**")
        st.dataframe(df.describe().round(2))


def _record_form(bundle: dict) -> tuple[bool, dict, bool]:
    cats = bundle["categories"]
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        flight_id = c1.text_input("Flight ID (optional)", placeholder="e.g. FL1001")
        airline = c2.selectbox("Airline", cats["airlines"])
        weather = c3.selectbox("Weather condition", WEATHER_CONDITIONS)

        c1, c2, c3 = st.columns(3)
        origin = c1.selectbox("Origin airport", cats["airports"], index=0)
        destination = c2.selectbox("Destination airport", cats["airports"],
                                   index=min(1, len(cats["airports"]) - 1))
        distance = c3.number_input("Flight distance (km)", value=1500.0, step=50.0)

        c1, c2, c3 = st.columns(3)
        date = c1.date_input("Scheduled departure date", value=dt.date.today())
        dep_time = c2.time_input("Scheduled departure time", value=dt.time(9, 0), step=300)
        arr_time = c3.time_input("Scheduled arrival time", value=dt.time(11, 30), step=300)

        c1, c2, c3, c4 = st.columns(4)
        temperature = c1.number_input("Temperature (°C)", value=15.0, step=1.0)
        wind = c2.number_input("Wind speed (km/h)", value=12.0, step=1.0)
        visibility = c3.number_input("Visibility (km)", value=10.0, step=0.5)
        traffic = c4.selectbox("Traffic level", TRAFFIC_LEVELS, index=1)

        st.markdown("**Previous delay information**")
        c1, c2 = st.columns([1, 2])
        prev_known = c1.checkbox("Previous leg's delay is known before departure", value=False)
        prev_delay = c2.number_input("Previous delay (minutes)", value=0.0, step=5.0,
                                     help="Only used if the box on the left is ticked.")
        save = st.checkbox("Save this prediction to local history", value=True)
        submitted = st.form_submit_button("Predict Flight Delay", type="primary")

    record = {
        "Airline": airline, "Origin": origin, "Destination": destination,
        "Departure_Date": date.isoformat() if date else None,
        "Scheduled_Departure": dep_time.strftime("%H:%M") if dep_time else None,
        "Scheduled_Arrival": arr_time.strftime("%H:%M") if arr_time else None,
        "Distance_km": distance, "Weather_Condition": weather, "Temperature_C": temperature,
        "Wind_Speed_kmh": wind, "Visibility_km": visibility, "Traffic_Level": traffic,
        "Previous_Delay_Min": prev_delay if prev_known else None,
    }
    if flight_id.strip():
        record["Flight_ID"] = flight_id.strip()
    return submitted, record, save


def page_predict() -> None:
    hero("Flight Delay Prediction", "Enter flight details known before departure.")
    bundle = require_model()
    if bundle is None:
        return

    with st.expander("Risk category thresholds (configurable)"):
        t1, t2 = st.columns(2)
        low = t1.slider("Low risk below", 0.05, 0.90, DEFAULT_RISK_THRESHOLDS["low_max"], 0.05)
        high = t2.slider("High risk at or above", 0.10, 0.95, DEFAULT_RISK_THRESHOLDS["high_min"], 0.05)
        st.caption("Low: probability < low threshold · Medium: in between · High: probability ≥ high threshold.")
    thresholds = {"low_max": low, "high_min": high}
    if low >= high:
        st.error("The 'Low risk below' threshold must be smaller than the 'High risk at or above' threshold.")
        return

    submitted, record, save = _record_form(bundle)
    if submitted:
        try:
            st.session_state["last_prediction"] = predict_single(bundle, record, thresholds)
            st.session_state["last_thresholds"] = thresholds
            if save:
                result_row = {k: v for k, v in st.session_state["last_prediction"].items() if k != "Explanation"}
                append_history(build_history_records(pd.DataFrame([result_row])))
        except InputValidationError as exc:
            st.session_state.pop("last_prediction", None)
            st.error("Please fix the following before predicting:")
            for message in exc.messages:
                st.markdown(f"- {message}")

    result = st.session_state.get("last_prediction")
    if not result:
        return

    st.divider()
    delayed = result["Predicted_Status"] == "Delayed"
    badge = "badge-delayed" if delayed else "badge-ontime"
    st.markdown(f'<span class="badge {badge}">{result["Predicted_Status"]}</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of delay", f"{result['Delay_Probability']:.1%}")
    c2.metric("Risk category", result["Risk_Category"])
    est = result.get("Estimated_Delay_Min")
    c3.metric("Estimated delay", f"{est:.0f} min" if est is not None and pd.notna(est) else "n/a")

    left, right = st.columns([1, 1])
    with left:
        show_chart(viz.risk_gauge(result["Delay_Probability"], st.session_state.get("last_thresholds")))
    with right:
        st.subheader("Why this prediction?")
        factors = result.get("Explanation", [])
        if factors:
            for f in factors[:3]:
                st.markdown(
                    f"- **{f['factor']}** ({f['value']}, typical: {f['typical_value']}) "
                    f"{f['direction']} the odds of delay by about ×{f['odds_multiplier']:.2f}"
                    if f["direction"] == "raises" else
                    f"- **{f['factor']}** ({f['value']}, typical: {f['typical_value']}) "
                    f"{f['direction']} the odds of delay (×{f['odds_multiplier']:.2f})"
                )
        else:
            st.write("All inputs are close to typical values, so no single factor stands out.")
    if factors:
        show_chart(viz.factor_impact_chart(factors))

    st.caption(
        "The probability is a model estimate based on patterns in the training data, not a guarantee. "
        "Factor effects compare your inputs with a typical training flight and describe the model's "
        "behaviour, not proven causes. The estimated delay is non-negative (early departures count as 0)."
    )


def _template_csv() -> bytes:
    sample = pd.DataFrame([
        {"Flight_ID": "FL1001", "Airline": "SkyJet Airways", "Origin": "JFK", "Destination": "ORD",
         "Departure_Date": "2026-12-05", "Scheduled_Departure": "18:30", "Scheduled_Arrival": "20:10",
         "Distance_km": 1190, "Weather_Condition": "Rain", "Temperature_C": 4.0, "Wind_Speed_kmh": 22.0,
         "Visibility_km": 6.0, "Traffic_Level": "High", "Previous_Delay_Min": 20},
        {"Flight_ID": "FL1002", "Airline": "BlueHorizon Air", "Origin": "SFO", "Destination": "DEN",
         "Departure_Date": "2026-12-06", "Scheduled_Departure": "07:15", "Scheduled_Arrival": "10:00",
         "Distance_km": 1630, "Weather_Condition": "Clear", "Temperature_C": 8.0, "Wind_Speed_kmh": 9.0,
         "Visibility_km": 10.0, "Traffic_Level": "Low", "Previous_Delay_Min": ""},
    ])
    return sample.to_csv(index=False).encode("utf-8")


def page_batch() -> None:
    hero("Batch Prediction", "Upload a CSV of flights and download the predictions.")
    bundle = require_model()
    if bundle is None:
        return

    st.markdown(
        "**Required columns:** " + ", ".join(f"`{c}`" for c in REQUIRED_INPUT_COLUMNS) +
        "  \n**Optional:** `Flight_ID`, `Previous_Delay_Min` (only if known before departure). "
        "Actual departure times and delay minutes are **not** needed and are ignored."
    )
    st.download_button("Download CSV template", _template_csv(), file_name="flight_input_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload flights CSV", type=["csv"])
    save = st.checkbox("Add these predictions to local history", value=False)
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty.")
        return
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"The file could not be read as a CSV: {exc}")
        return
    if df.empty:
        st.error("The file has a header but no data rows.")
        return
    if len(df) > MAX_BATCH_ROWS:
        st.error(f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,}.")
        return

    try:
        results, errors = predict_dataframe(bundle, df)
    except MissingColumnsError as exc:
        st.error(str(exc))
        st.write("Columns found in your file:", ", ".join(map(str, df.columns)))
        return

    delayed = int((results["Predicted_Status"] == "Delayed").sum()) if len(results) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows uploaded", len(df))
    c2.metric("Predicted delayed", delayed)
    c3.metric("Predicted on-time", len(results) - delayed)
    c4.metric("Rows rejected", len(errors))

    if len(errors):
        st.warning(f"{len(errors)} row(s) had missing or invalid values and were skipped.")
        show_table(errors)
    if results.empty:
        st.error("No valid rows to predict.")
        return

    st.subheader("Predictions")
    show_table(results)
    st.download_button("Download predictions (flight_delay_predictions.csv)",
                       results.to_csv(index=False).encode("utf-8"),
                       file_name="flight_delay_predictions.csv", mime="text/csv", type="primary")
    if save and st.button("Confirm: save to history"):
        count = append_history(build_history_records(results))
        st.success(f"Saved {count} prediction(s) to local history.")


def page_analytics() -> None:
    hero("Analytics", "Delay patterns in the dataset.")
    df = require_dataset()
    if df is None:
        return
    st.caption(SYNTHETIC_NOTE)
    st.warning("These charts show **associations** in the dataset. They do not establish that any "
               "factor *causes* delays.")

    a, b = st.columns(2)
    with a:
        show_chart(viz.status_chart(df))
    with b:
        show_chart(viz.delay_rate_chart(df, "Airline", "Delay percentage by airline"))
    show_chart(viz.delay_by_hour_chart(df))
    a, b = st.columns(2)
    with a:
        show_chart(viz.delay_rate_chart(df, "Weather_Condition", "Delay patterns by weather", WEATHER_CONDITIONS))
    with b:
        show_chart(viz.delay_rate_chart(df, "Traffic_Level", "Delay patterns by traffic level", TRAFFIC_LEVELS))
    show_chart(viz.delay_distribution_chart(df))
    show_chart(viz.origin_destination_heatmap(df))


def page_performance() -> None:
    hero("Model Performance", "All numbers below come from the saved evaluation results.")
    metrics = get_metrics()
    if metrics is None:
        st.error("Evaluation metrics not found (models/model_metrics.json).")
        st.code("python generate_dataset.py\npython train_model.py", language="bash")
        return

    sel = metrics["selection"]["selected_model"]
    split = metrics["split"]
    st.success(f"Selected model: **{sel}**")
    st.markdown(f"**Selection criterion:** {metrics['selection']['criterion']}")
    st.caption(f"Train rows: {split['train_rows']} · Test rows: {split['test_rows']} (stratified) · "
               f"Decision threshold: {metrics['selection']['decision_threshold']} · {metrics['dataset_note']}")

    rows = []
    for name, res in metrics["models"].items():
        t = res["test"]
        rows.append({
            "Model": name + (" ✓" if name == sel else ""),
            "CV F1 (train)": round(res["cv_f1_mean"], 3),
            "Accuracy": round(t["accuracy"], 3), "Precision": round(t["precision"], 3),
            "Recall": round(t["recall"], 3), "F1": round(t["f1"], 3),
            "ROC-AUC": round(t["roc_auc"], 3) if t["roc_auc"] is not None else None,
        })
    show_table(pd.DataFrame(rows))
    show_chart(viz.model_comparison_chart(metrics))

    choice = st.selectbox("Inspect model", list(metrics["models"]), index=list(metrics["models"]).index(sel))
    a, b = st.columns(2)
    with a:
        show_chart(viz.confusion_matrix_chart(metrics["models"][choice]["test"]["confusion_matrix"],
                                              f"Confusion matrix - {choice}"))
    with b:
        roc = viz.roc_chart(metrics, highlight=choice)
        if roc is not None:
            show_chart(roc)
        else:
            st.info("ROC curve unavailable: the test set does not contain both classes.")

    st.subheader(f"Feature importance - {sel}")
    importance = metrics.get("feature_importance")
    if importance:
        st.caption(importance["type"])
        t1, t2 = st.tabs(["By input", "Top encoded features"])
        with t1:
            show_chart(viz.feature_importance_chart(importance["grouped"], "Importance by input (normalised)"))
        with t2:
            show_chart(viz.feature_importance_chart(importance["detail"], "Top encoded features"))
    else:
        st.info("Feature importance is not available for this model type.")

    reg = metrics.get("regression")
    if reg:
        st.subheader("Delay-minutes regression model")
        r1, r2, r3 = st.columns(3)
        r1.metric("MAE (min)", f"{reg['mae']:.2f}")
        r2.metric("RMSE (min)", f"{reg['rmse']:.2f}")
        r3.metric("MAE of 'always predict mean'", f"{reg['baseline_mae_predict_mean']:.2f}")
        st.caption(reg["note"])

    st.subheader("Limitations")
    for item in metrics["limitations"]:
        st.markdown(f"- {item}")


def page_history() -> None:
    hero("Prediction History", "Predictions saved on this computer.")
    st.info(f"This is a local application. History is stored only in `{HISTORY_PATH.name}` inside the "
            "project's data folder and contains no names or contact details.")
    history = load_history()
    if history.empty:
        st.write("No predictions saved yet.")
        return
    st.write(f"**{len(history)}** saved prediction(s), newest first.")
    show_table(history)
    st.download_button("Download history (CSV)", history.to_csv(index=False).encode("utf-8"),
                       file_name="prediction_history.csv", mime="text/csv")
    confirm = st.checkbox("I want to permanently delete all saved history")
    if st.button("Clear history", disabled=not confirm):
        clear_history()
        st.success("History cleared.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
def main() -> None:
    st.sidebar.title("✈️ Flight Delay Detection")
    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.divider()
    st.sidebar.caption("Machine-learning estimates only. Not a live flight-status service. "
                       "Bundled data is synthetic.")
    {
        "Home Dashboard": page_home,
        "Dataset Explorer": page_explorer,
        "Flight Delay Prediction": page_predict,
        "Batch Prediction": page_batch,
        "Analytics": page_analytics,
        "Model Performance": page_performance,
        "Prediction History": page_history,
    }[page]()


main()
