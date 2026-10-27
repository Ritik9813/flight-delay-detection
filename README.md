# ✈️ Flight Delay Detection and Prediction System

An end-to-end machine-learning project: a **synthetic 500-flight dataset**, a training script that compares
three classifiers (plus a delay-minutes regressor), and a **Streamlit dashboard** for single and batch
predictions, analytics, model evaluation and prediction history.

> **Important**
> * This is a **prediction model, not a live flight-tracking service.** It cannot tell you the real-time
>   status of any real flight.
> * `data/flights_500.csv` is **synthetic** (simulated records, fictional airlines). It is *not* actual
>   airline data, and the model's scores must **not** be read as real-world airline performance.

---

## 1. Quick start

Requires **Python 3.9+**. Run all commands from the project folder (`flight-delay-detection/`).

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # in Command Prompt: .venv\Scripts\activate.bat
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```
If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

The app opens at <http://localhost:8501>. The repository already ships with a generated dataset and trained
model, but a model must be trained with the same scikit-learn version you run it with, so **run
`python train_model.py` once after installing**. (The app tells you if it's needed.)

### Notebook version
Open `flight_delay_detection.ipynb` in Jupyter / VS Code / Colab and choose *Run All*. It is self-contained
(generates data, trains, evaluates, charts, predicts) and trains in your own environment, so it never has a
scikit-learn version mismatch.

### Run the tests
```bash
python -m pytest -q
```

---

## 2. Workflow

1. **Generate data** - `python generate_dataset.py` writes exactly 500 synthetic flights to `data/flights_500.csv`
   (options: `--rows`, `--seed`, `--output`).
2. **Train** - `python train_model.py` cleans the data, splits it, compares the models, selects the best, trains
   the regressor and saves `models/flight_delay_model.joblib` and `models/model_metrics.json`.
3. **Launch** - `streamlit run app.py`. The app only *loads* the saved model; it never retrains. If the
   artifacts are missing it shows instructions instead of crashing.
4. Predict one flight, or upload a CSV, then review analytics and model performance.

## 3. Dashboard pages

| Page | What it does |
|---|---|
| Home Dashboard | Totals, delayed/on-time counts, delay %, selected model and its test F1 |
| Dataset Explorer | Filter/search table, summary stats, missing values and duplicates, CSV download |
| Flight Delay Prediction | Validated form → status, delay probability, risk category, estimated minutes, key factors |
| Batch Prediction | Upload CSV → validation report, predictions, counts, download `flight_delay_predictions.csv` |
| Analytics | Plotly charts: status, airline, hour, weather, traffic, delay distribution, route heat-map |
| Model Performance | Model comparison, confusion matrix, ROC, feature importance, regression MAE/RMSE, limitations |
| Prediction History | Local log of predictions with a clear-history button |

All metrics on the Model Performance page are read from `models/model_metrics.json`, which is written by
`train_model.py` from real evaluation results.

### Batch CSV format
Required columns: `Airline, Origin, Destination, Departure_Date (YYYY-MM-DD), Scheduled_Departure (HH:MM),
Scheduled_Arrival (HH:MM), Distance_km, Weather_Condition, Temperature_C, Wind_Speed_kmh, Visibility_km,
Traffic_Level`. Optional: `Flight_ID`, `Previous_Delay_Min` (only if known **before** departure).
`Weather_Condition` ∈ Clear, Cloudy, Rain, Fog, Snow, Thunderstorm; `Traffic_Level` ∈ Low, Medium, High.
Download a ready-made template from the Batch Prediction page. Actual departure / delay columns are not
needed and are ignored. Invalid rows are skipped and listed with the reason.

---

## 4. How the system works

**Target.** `Delayed` = departure delay ≥ 15 minutes, otherwise `On-Time`. The label is always derived from
`Departure_Delay_Min` (the training code recomputes it and reports any inconsistency).

**No target leakage.** `Actual_Departure`, `Departure_Delay_Min` and `Delay_Status` are never model inputs.
Inputs are only things knowable before departure: airline, route, schedule, distance, weather, traffic and
(optionally) the previous leg's delay. Tests assert this.

**One pipeline for training and inference** (`src/preprocessing.py`): feature engineering
(departure hour, day of week, month, weekend, peak-hour, scheduled duration incl. overnight flights) →
`ColumnTransformer` (median-impute + scale numerics; most-frequent-impute + one-hot categoricals, unseen
categories ignored) → estimator, all inside a single scikit-learn `Pipeline` saved with Joblib.
Imputers are fitted on training data only. `Previous_Delay_Known` is an explicit indicator, so a blank previous
delay is handled consistently.

**Models and selection.** Logistic Regression, Random Forest and Gradient Boosting are compared. The stratified
80/20 split is fixed by a random seed. The model is **selected by highest mean 5-fold cross-validated F1 on the
training set**; F1 balances precision and recall, so it is not fooled by class imbalance the way accuracy can
be. The test set is only used for the final report, so selection does not leak into the test metrics.
Reported: accuracy, precision, recall, F1, confusion matrix, ROC-AUC.

**Delay-minutes regressor.** A Gradient Boosting regressor predicts minutes. Early departures (negative delay)
are treated as 0 and predictions are clipped to be non-negative. Evaluated with MAE and RMSE and compared to
an "always predict the mean" baseline.

**Risk categories.** Based on the predicted probability: Low `< 0.30`, Medium in between, High `≥ 0.60`.
Thresholds are configurable in the prediction page (defaults in `src/config.py`).

**Explanations.** For each key input, the app swaps it for the typical training value and measures how the
model's odds of delay change. This describes the *model's behaviour*; it does not prove causes.

**Privacy.** Prediction history is a local CSV (`data/prediction_history.csv`) with timestamp, optional flight ID,
a short input summary and the result - no names or contact details. It can be cleared from the History page.

---

## 5. Project structure

```
flight-delay-detection/
├── app.py                  Streamlit dashboard (7 pages)
├── train_model.py          Train, evaluate, select, save
├── generate_dataset.py     Synthetic 500-flight generator
├── requirements.txt
├── README.md
├── flight_delay_detection.ipynb   Self-contained notebook of the whole workflow
├── data/flights_500.csv
├── models/
│   ├── flight_delay_model.joblib
│   └── model_metrics.json
├── src/
│   ├── config.py           Paths, column lists, thresholds
│   ├── preprocessing.py    Validation, cleaning, features, pipeline
│   ├── prediction.py       Model loading, single/batch prediction, explanations
│   ├── evaluation.py       Metrics, model selection, feature importance
│   ├── history.py          Local prediction history
│   └── visualization.py    Plotly charts
└── tests/                  52 pytest tests (preprocessing, prediction, app smoke test)
```

## 6. Limitations

* **Small data:** 500 records (about 100 in the test set). Metrics are noisy and shift with a different split;
  model ranking by cross-validation and by the test set can disagree, as it may in your run.
* **Synthetic data:** the delay relationships were written into the generator. A good score shows the model can
  recover simulated rules, not that it can predict real airline operations.
* **Missing real-world drivers:** crew, maintenance, air-traffic control, airport slots and cascading
  network effects are not modelled.
* Probabilities are uncalibrated estimates, not guarantees.
* Not a live service: no flight API is integrated.
* The deployed classifier is trained on the training split only (so the reported test metrics describe exactly
  the saved model); a production system would refit on all data after validation.
* Pickled models depend on the scikit-learn version. The app detects a mismatch and asks you to run
  `python train_model.py` on your machine (do this once after installing, e.g. `AttributeError: ... 'multi_class'`).

## 7. Possible future enhancements

* Use real historical data (for example public on-time-performance records) and re-validate time-based
  (train on earlier dates, test on later dates).
* Integrate live weather and flight-schedule APIs for genuine real-time inputs.
* Probability calibration, threshold tuning for a chosen cost of false alarms vs misses, hyper-parameter search.
* SHAP-based explanations, model monitoring and drift detection, and a REST API around the model.
* Cascading-delay features (aircraft rotation, turnaround time) and airport-level congestion indices.
