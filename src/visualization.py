"""Plotly figure builders used by the dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import DEFAULT_RISK_THRESHOLDS, DELAY_THRESHOLD_MIN
from src.preprocessing import add_scheduled_hour

NAVY = "#0B2545"
SKY = "#1F78C8"
TEAL = "#2A9D8F"
ALERT = "#E4572E"
AMBER = "#F2A541"
GRID = "#E3EAF3"
STATUS_COLORS = {"On-Time": TEAL, "Delayed": ALERT}


def _style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=NAVY)) if title else None,
        height=height,
        margin=dict(l=30, r=20, t=60 if title else 20, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _delay_rate_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    g = df.groupby(column, observed=True)["Delay_Status"]
    table = pd.DataFrame({
        "Flights": g.size(),
        "Delayed": g.apply(lambda s: int((s == "Delayed").sum())),
    }).reset_index()
    table["Delay_Rate_%"] = (table["Delayed"] / table["Flights"] * 100).round(1)
    return table


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def status_chart(df: pd.DataFrame) -> go.Figure:
    counts = df["Delay_Status"].value_counts().reindex(["On-Time", "Delayed"]).fillna(0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.55,
        marker=dict(colors=[STATUS_COLORS[k] for k in counts.index]),
        textinfo="label+percent",
    ))
    return _style(fig, "Delayed vs on-time flights", 340)


def delay_rate_chart(df: pd.DataFrame, column: str, title: str, order: list | None = None) -> go.Figure:
    table = _delay_rate_table(df, column)
    if order:
        table[column] = pd.Categorical(table[column], categories=order, ordered=True)
        table = table.sort_values(column)
    else:
        table = table.sort_values("Delay_Rate_%", ascending=False)
    fig = px.bar(
        table, x=column, y="Delay_Rate_%", text="Delay_Rate_%",
        hover_data={"Flights": True, "Delayed": True},
        color="Delay_Rate_%", color_continuous_scale=[TEAL, AMBER, ALERT],
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="Delayed flights (%)")
    fig.update_yaxes(range=[0, max(100, table["Delay_Rate_%"].max() * 1.15)])
    return _style(fig, title)


def delay_by_hour_chart(df: pd.DataFrame) -> go.Figure:
    work = add_scheduled_hour(df).dropna(subset=["Dep_Hour"])
    work["Dep_Hour"] = work["Dep_Hour"].astype(int)
    table = _delay_rate_table(work, "Dep_Hour")
    fig = go.Figure()
    fig.add_bar(x=table["Dep_Hour"], y=table["Flights"], name="Flights", marker_color=GRID, yaxis="y2")
    fig.add_scatter(
        x=table["Dep_Hour"], y=table["Delay_Rate_%"], name="Delay rate (%)",
        mode="lines+markers", line=dict(color=ALERT, width=3),
    )
    fig.update_layout(
        xaxis=dict(title="Scheduled departure hour", dtick=1),
        yaxis=dict(title="Delayed flights (%)", range=[0, 105]),
        yaxis2=dict(title="Flights", overlaying="y", side="right", showgrid=False),
    )
    return _style(fig, "Delay rate by scheduled departure hour")


def delay_distribution_chart(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Departure_Delay_Min", color="Delay_Status", nbins=40,
        color_discrete_map=STATUS_COLORS, barmode="overlay", opacity=0.8,
    )
    fig.add_vline(x=DELAY_THRESHOLD_MIN, line_dash="dash", line_color=NAVY,
                  annotation_text=f"{DELAY_THRESHOLD_MIN} min threshold")
    fig.update_layout(xaxis_title="Departure delay (minutes; negative = early)", yaxis_title="Flights")
    return _style(fig, "Delay duration distribution")


def origin_destination_heatmap(df: pd.DataFrame) -> go.Figure:
    work = df.assign(Delayed=(df["Delay_Status"] == "Delayed").astype(int))
    rate = work.pivot_table(index="Origin", columns="Destination", values="Delayed", aggfunc="mean") * 100
    count = work.pivot_table(index="Origin", columns="Destination", values="Delayed", aggfunc="size")
    fig = go.Figure(go.Heatmap(
        z=rate.values, x=list(rate.columns), y=list(rate.index),
        customdata=count.reindex(index=rate.index, columns=rate.columns).values,
        colorscale=[[0, TEAL], [0.5, AMBER], [1, ALERT]], zmin=0, zmax=100,
        hovertemplate="%{y} → %{x}<br>Delay rate: %{z:.0f}%<br>Flights: %{customdata}<extra></extra>",
        colorbar=dict(title="Delay %"),
    ))
    fig.update_layout(xaxis_title="Destination", yaxis_title="Origin")
    return _style(fig, "Delay rate by route (origin → destination)", 440)


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
def model_comparison_chart(metrics: dict) -> go.Figure:
    rows = []
    for name, res in metrics["models"].items():
        for key in ("accuracy", "precision", "recall", "f1"):
            rows.append({"Model": name, "Metric": key.upper() if key == "f1" else key.title(),
                         "Score": res["test"][key]})
    fig = px.bar(pd.DataFrame(rows), x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_sequence=[SKY, TEAL, AMBER])
    fig.update_yaxes(range=[0, 1])
    return _style(fig, "Test-set metrics by model")


def confusion_matrix_chart(cm: list[list[int]], title: str = "Confusion matrix") -> go.Figure:
    labels = ["On-Time", "Delayed"]
    fig = px.imshow(
        np.array(cm), x=[f"Pred {l}" for l in labels], y=[f"Actual {l}" for l in labels],
        text_auto=True, color_continuous_scale=["#EAF2FB", SKY], aspect="auto",
    )
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, title, 340)


def roc_chart(metrics: dict, highlight: str | None = None) -> go.Figure | None:
    fig = go.Figure()
    drawn = False
    palette = [SKY, TEAL, AMBER]
    for i, (name, res) in enumerate(metrics["models"].items()):
        curve = res["test"].get("roc_curve")
        if not curve:
            continue
        drawn = True
        auc = res["test"]["roc_auc"]
        fig.add_scatter(
            x=curve["fpr"], y=curve["tpr"], mode="lines",
            name=f"{name} (AUC {auc:.2f})",
            line=dict(color=palette[i % 3], width=4 if name == highlight else 2),
        )
    if not drawn:
        return None
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                    line=dict(color="#9AA7B8", dash="dash"))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    return _style(fig, "ROC curves (test set)", 380)


def feature_importance_chart(items: list[dict], title: str) -> go.Figure:
    df = pd.DataFrame(items).sort_values("importance")
    fig = px.bar(df, x="importance", y="feature", orientation="h", color_discrete_sequence=[SKY])
    fig.update_layout(xaxis_title="Importance", yaxis_title=None)
    return _style(fig, title, max(320, 28 * len(df) + 100))


# --------------------------------------------------------------------------- #
# Prediction page
# --------------------------------------------------------------------------- #
def risk_gauge(probability: float, thresholds: dict | None = None) -> go.Figure:
    t = thresholds or DEFAULT_RISK_THRESHOLDS
    low, high = t["low_max"] * 100, t["high_min"] * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number=dict(suffix="%", font=dict(color=NAVY)),
        title=dict(text="Estimated probability of delay"),
        gauge=dict(
            axis=dict(range=[0, 100]), bar=dict(color=NAVY),
            steps=[
                dict(range=[0, low], color="#BFE6DF"),
                dict(range=[low, high], color="#FBE0B5"),
                dict(range=[high, 100], color="#F6B9A8"),
            ],
        ),
    ))
    return _style(fig, None, 260)


def factor_impact_chart(factors: list[dict]) -> go.Figure:
    df = pd.DataFrame(factors).sort_values("log_odds_shift")
    df["label"] = df["factor"] + ": " + df["value"]
    fig = go.Figure(go.Bar(
        x=df["log_odds_shift"], y=df["label"], orientation="h",
        marker_color=[ALERT if v > 0 else TEAL for v in df["log_odds_shift"]],
        customdata=df[["odds_multiplier", "typical_value"]].values,
        hovertemplate="%{y}<br>Odds ×%{customdata[0]} vs typical (%{customdata[1]})<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Shift in log-odds of delay vs a typical flight (right = more delay risk)")
    return _style(fig, "Key input factors", max(260, 55 * len(df) + 110))
