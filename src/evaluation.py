"""Metric calculation, model selection and feature-importance extraction."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.config import CATEGORICAL_FEATURES


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute every classification metric from real predictions.

    ``roc_auc`` and the ROC curve are ``None`` unless both classes are present
    in ``y_true`` and probability scores were supplied.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "support": {"on_time": int((y_true == 0).sum()), "delayed": int((y_true == 1).sum())},
        "roc_curve": None,
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        proba = np.asarray(y_proba, dtype=float)
        result["roc_auc"] = float(roc_auc_score(y_true, proba))
        fpr, tpr, _ = roc_curve(y_true, proba)
        result["roc_curve"] = {
            "fpr": np.round(fpr, 4).tolist(),
            "tpr": np.round(tpr, 4).tolist(),
        }
    return result


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def select_best_model(results: dict) -> str:
    """Pick the model with the highest mean cross-validated F1 on TRAINING data.

    Ties are broken by cross-validated ROC-AUC. The held-out test set is never
    used for selection, only for the final report.
    """
    if not results:
        raise ValueError("No model results to select from.")
    return max(
        results,
        key=lambda name: (results[name]["cv_f1_mean"], results[name].get("cv_roc_auc_mean") or 0.0),
    )


def _clean_name(raw: str) -> str:
    return raw.split("__", 1)[1] if "__" in raw else raw


def _group_name(feature: str) -> str:
    for col in CATEGORICAL_FEATURES:
        if feature.startswith(col + "_"):
            return col
    return feature


def extract_feature_importance(pipeline, top_n: int = 15) -> dict | None:
    """Return importances for tree models (``feature_importances_``) or linear
    models (absolute coefficients on standardised inputs). ``None`` if unsupported."""
    model = pipeline.named_steps["model"]
    names = [_clean_name(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "Impurity-based importance"
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        kind = "Absolute coefficient (standardised inputs)"
    else:
        return None

    detail = sorted(zip(names, values), key=lambda t: t[1], reverse=True)
    grouped: dict[str, float] = {}
    for name, value in detail:
        grouped[_group_name(name)] = grouped.get(_group_name(name), 0.0) + float(value)
    total = sum(grouped.values()) or 1.0
    grouped_list = sorted(
        ({"feature": k, "importance": v / total} for k, v in grouped.items()),
        key=lambda d: d["importance"], reverse=True,
    )
    return {
        "type": kind,
        "detail": [{"feature": n, "importance": float(v)} for n, v in detail[:top_n]],
        "grouped": grouped_list,
    }
