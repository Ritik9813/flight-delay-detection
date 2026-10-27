"""Central configuration: paths, column definitions and modelling constants."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_PATH = DATA_DIR / "flights_500.csv"
MODEL_PATH = MODELS_DIR / "flight_delay_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
HISTORY_PATH = DATA_DIR / "prediction_history.csv"

RANDOM_STATE = 42

# A flight is "Delayed" when its departure delay is at least this many minutes.
DELAY_THRESHOLD_MIN = 15
# Probability at/above which the classifier labels a flight "Delayed".
DECISION_THRESHOLD = 0.5
# Configurable risk bands (probability of delay). Low: p < low_max,
# High: p >= high_min, Medium: everything in between.
DEFAULT_RISK_THRESHOLDS = {"low_max": 0.30, "high_min": 0.60}

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Fog", "Snow", "Thunderstorm"]
TRAFFIC_LEVELS = ["Low", "Medium", "High"]

# Columns that must exist in any input used for prediction (single or batch).
REQUIRED_INPUT_COLUMNS = [
    "Airline", "Origin", "Destination", "Departure_Date",
    "Scheduled_Departure", "Scheduled_Arrival", "Distance_km",
    "Weather_Condition", "Temperature_C", "Wind_Speed_kmh",
    "Visibility_km", "Traffic_Level",
]
# Optional at prediction time. Previous_Delay_Min is only usable when the
# previous leg's delay is genuinely known before the flight departs.
OPTIONAL_INPUT_COLUMNS = ["Flight_ID", "Previous_Delay_Min"]
INPUT_FEATURE_COLUMNS = REQUIRED_INPUT_COLUMNS + ["Previous_Delay_Min"]

# Columns that describe the OUTCOME and must never be used as model inputs.
LEAKAGE_COLUMNS = ["Actual_Departure", "Departure_Delay_Min", "Delay_Status"]

# Dataset columns (training file).
DATASET_COLUMNS = [
    "Flight_ID", "Airline", "Origin", "Destination", "Departure_Date",
    "Scheduled_Departure", "Scheduled_Arrival", "Distance_km",
    "Weather_Condition", "Temperature_C", "Wind_Speed_kmh", "Visibility_km",
    "Traffic_Level", "Previous_Delay_Min", "Actual_Departure",
    "Departure_Delay_Min", "Delay_Status",
]

# Engineered feature groups fed to the ColumnTransformer.
CATEGORICAL_FEATURES = ["Airline", "Origin", "Destination", "Weather_Condition", "Traffic_Level"]
NUMERIC_FEATURES = [
    "Distance_km", "Temperature_C", "Wind_Speed_kmh", "Visibility_km",
    "Previous_Delay_Min", "Previous_Delay_Known", "Dep_Hour", "Day_Of_Week",
    "Month", "Is_Weekend", "Is_Peak_Hour", "Scheduled_Duration_Min",
]
