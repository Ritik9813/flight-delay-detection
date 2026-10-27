"""Generate a SYNTHETIC flight dataset (default: exactly 500 records).

The records are simulated. They are NOT actual airline data, and the airline
names are fictional. The target label is derived from the generated delay
minutes (Delayed = delay >= 15 minutes), never generated independently.

Usage:
    python generate_dataset.py
    python generate_dataset.py --rows 500 --seed 42 --output data/flights_500.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PATH, DATASET_COLUMNS, DELAY_THRESHOLD_MIN, RANDOM_STATE

# Real airport codes with approximate coordinates (used only for distances).
AIRPORTS = {
    "JFK": (40.64, -73.78), "LAX": (33.94, -118.41), "ORD": (41.98, -87.90),
    "ATL": (33.64, -84.43), "DFW": (32.90, -97.04), "DEN": (39.86, -104.67),
    "SFO": (37.62, -122.38), "MIA": (25.79, -80.29),
}
# Fictional airlines with an invented "operational reliability" offset (minutes).
AIRLINES = {
    "SkyJet Airways": 2.0, "BlueHorizon Air": -2.5, "Meridian Airlines": 0.5,
    "Aurora Air": 4.0, "Pacific Wing": -1.0,
}
AIRLINE_WEIGHTS = [0.26, 0.22, 0.20, 0.16, 0.16]
# Extra delay minutes at congested origin airports.
AIRPORT_CONGESTION = {"JFK": 3.0, "ORD": 4.0, "ATL": 2.0, "SFO": 2.5}

WEATHER_EFFECT = {"Clear": 0.0, "Cloudy": 1.5, "Rain": 6.0, "Fog": 12.0, "Snow": 15.0, "Thunderstorm": 21.0}
TRAFFIC_EFFECT = {"Low": 0.0, "Medium": 4.0, "High": 11.0}


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (*a, *b))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(h)))


def generate_dataset(n_rows: int = 500, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create ``n_rows`` plausible synthetic flights with delay-driven labels."""
    rng = np.random.default_rng(seed)
    codes = list(AIRPORTS)
    airline_names = list(AIRLINES)
    rows = []

    start = pd.Timestamp("2025-01-01")
    for i in range(n_rows):
        origin = rng.choice(codes)
        destination = rng.choice([c for c in codes if c != origin])
        airline = rng.choice(airline_names, p=AIRLINE_WEIGHTS)
        date = start + pd.Timedelta(days=int(rng.integers(0, 365)))
        month = date.month

        hour_weights = np.array([4, 8, 9, 8, 6, 5, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 3, 2, 2], dtype=float)
        hour = int(rng.choice(np.arange(5, 24), p=hour_weights / hour_weights.sum()))
        minute = int(rng.choice(np.arange(0, 60, 5)))
        dep_dt = date + pd.Timedelta(hours=hour, minutes=minute)

        distance = round(_haversine_km(AIRPORTS[origin], AIRPORTS[destination]) * 1.05)
        duration_min = int(round((distance / 800.0 * 60.0 + 30.0) / 5.0) * 5)
        arr_dt = dep_dt + pd.Timedelta(minutes=duration_min)

        temperature = 13 + 14 * np.sin(2 * np.pi * (month - 4) / 12) + rng.normal(0, 6)
        weather = rng.choice(
            list(WEATHER_EFFECT), p=[0.38, 0.27, 0.17, 0.07, 0.05, 0.06]
        )
        if weather == "Snow" and temperature > 3:
            weather = "Rain"
        if weather == "Thunderstorm" and temperature < 8:
            weather = "Rain"

        wind = rng.gamma(2.5, 7.0) + (14.0 if weather == "Thunderstorm" else 0.0)
        visibility = {
            "Fog": rng.uniform(0.5, 3.0), "Snow": rng.uniform(1.5, 6.0),
            "Rain": rng.uniform(4.0, 9.0), "Thunderstorm": rng.uniform(3.0, 8.0),
        }.get(weather, rng.uniform(8.0, 10.0))

        peak = (6 <= hour <= 9) or (16 <= hour <= 20)
        traffic = rng.choice(
            ["Low", "Medium", "High"], p=[0.15, 0.45, 0.40] if peak else [0.40, 0.42, 0.18]
        )

        # Delay carried over from the aircraft's previous leg (known before departure).
        if rng.random() < 0.55:
            prev = rng.normal(0, 4)
        else:
            prev = rng.exponential(22.0)
        prev = int(round(float(np.clip(prev + AIRLINES[airline] * 0.5, -10, 240))))

        mean_delay = (
            -9.0
            + WEATHER_EFFECT[weather]
            + TRAFFIC_EFFECT[traffic]
            + 0.45 * max(prev, 0)
            + 0.5 * max(wind - 25.0, 0.0)
            + 2.5 * max(5.0 - visibility, 0.0)
            + (5.0 if 17 <= hour <= 21 else 0.0)
            + (-3.0 if hour <= 7 else 0.0)
            + AIRLINES[airline]
            + AIRPORT_CONGESTION.get(origin, 0.0)
        )
        delay = mean_delay + rng.normal(0, 6.0) + rng.exponential(4.0) - 4.0
        delay = int(round(float(np.clip(delay, -15, 360))))
        actual_dt = dep_dt + pd.Timedelta(minutes=delay)

        rows.append({
            "Flight_ID": f"FL{i + 1:04d}",
            "Airline": airline,
            "Origin": origin,
            "Destination": destination,
            "Departure_Date": date.strftime("%Y-%m-%d"),
            "Scheduled_Departure": dep_dt.strftime("%H:%M"),
            "Scheduled_Arrival": arr_dt.strftime("%H:%M"),
            "Distance_km": int(distance),
            "Weather_Condition": weather,
            "Temperature_C": round(float(temperature), 1),
            "Wind_Speed_kmh": round(float(wind), 1),
            "Visibility_km": round(float(visibility), 1),
            "Traffic_Level": traffic,
            "Previous_Delay_Min": prev,
            "Actual_Departure": actual_dt.strftime("%Y-%m-%d %H:%M"),
            "Departure_Delay_Min": delay,
            "Delay_Status": "Delayed" if delay >= DELAY_THRESHOLD_MIN else "On-Time",
        })

    df = pd.DataFrame(rows, columns=DATASET_COLUMNS)

    # Realistic data gaps (blank sensor readings / unknown previous-leg delay).
    for column, count in (("Temperature_C", 5), ("Visibility_km", 4), ("Previous_Delay_Min", 8)):
        if n_rows > 50:
            idx = rng.choice(n_rows, size=count, replace=False)
            df[column] = df[column].astype(float)
            df.loc[idx, column] = np.nan
    df["Previous_Delay_Min"] = df["Previous_Delay_Min"].astype("Float64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=DATA_PATH)
    args = parser.parse_args()

    df = generate_dataset(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    delayed = int((df["Delay_Status"] == "Delayed").sum())
    print(f"Saved {len(df)} SYNTHETIC flight records to {args.output}")
    print(f"Delayed: {delayed} ({delayed / len(df):.1%})  |  On-Time: {len(df) - delayed}")
    print("Reminder: this data is simulated and does not represent real airline performance.")


if __name__ == "__main__":
    main()
