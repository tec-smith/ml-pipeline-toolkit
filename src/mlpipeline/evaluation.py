"""Metrics on held-out data, a baseline to compare against, and the report."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as m
from sklearn.dummy import DummyClassifier, DummyRegressor

from mlpipeline.data import Schema

# Metrics where a lower value is better, for the report's comparison column.
LOWER_IS_BETTER = {"mae", "rmse", "median_ae", "mape", "log_loss"}


def evaluate(
    model: Any, X: pd.DataFrame, y: pd.Series, schema: Schema
) -> dict[str, Any]:
    """Score a fitted model on labelled data.

    Returns ``{"metrics": {...}}``, plus ``"confusion_matrix"`` and
    ``"per_class"`` for classification.
    """
    predictions = model.predict(X)
    if schema.task == "regression":
        return {"metrics": regression_metrics(y, predictions)}
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return classification_report(y, predictions, proba, schema)


def baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    schema: Schema,
) -> dict[str, float]:
    """Metrics of a model that ignores the features.

    The most frequent class for classification, the mean for regression. A
    model that can't beat this hasn't learned anything useful.
    """
    dummy = (
        DummyClassifier(strategy="prior")
        if schema.task == "classification"
        else DummyRegressor(strategy="mean")
    )
    dummy.fit(X_train, y_train)
    metrics: dict[str, float] = evaluate(dummy, X_test, y_test, schema)["metrics"]
    return metrics


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "mae": m.mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(m.mean_squared_error(y_true, y_pred)),
        "median_ae": m.median_absolute_error(y_true, y_pred),
        "r2": m.r2_score(y_true, y_pred),
    }
    # MAPE is meaningless when a true value is zero.
    if np.all(y_true != 0):
        out["mape"] = m.mean_absolute_percentage_error(y_true, y_pred)
    return {k: float(v) for k, v in out.items()}


def classification_report(
    y_true: Any, y_pred: Any, proba: np.ndarray | None, schema: Schema
) -> dict[str, Any]:
    classes = list(schema.classes or sorted(pd.unique(np.asarray(y_true))))
    binary = len(classes) == 2
    averaging: dict[str, Any] = (
        {"average": "binary", "pos_label": schema.positive_class}
        if binary
        else {"average": "macro"}
    )
    metrics: dict[str, float] = {
        "accuracy": m.accuracy_score(y_true, y_pred),
        "balanced_accuracy": m.balanced_accuracy_score(y_true, y_pred),
        "precision": m.precision_score(y_true, y_pred, zero_division=0, **averaging),
        "recall": m.recall_score(y_true, y_pred, zero_division=0, **averaging),
        "f1": m.f1_score(y_true, y_pred, zero_division=0, **averaging),
    }
    present = set(pd.unique(np.asarray(y_true)))
    if proba is not None and present == set(classes):
        metrics["log_loss"] = m.log_loss(y_true, proba, labels=classes)
        if binary:
            positive = classes.index(schema.positive_class)
            metrics["roc_auc"] = m.roc_auc_score(
                np.asarray(y_true) == schema.positive_class, proba[:, positive]
            )
        else:
            metrics["roc_auc"] = m.roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=classes
            )

    matrix = m.confusion_matrix(y_true, y_pred, labels=classes)
    precision, recall, f1, support = m.precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "confusion_matrix": {
            "labels": classes,
            "matrix": matrix.tolist(),
        },
        "per_class": {
            str(c): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, c in enumerate(classes)
        },
    }


def render_report(metadata: dict[str, Any]) -> str:
    """A Markdown summary of a run, saved next to the model as ``report.md``."""
    schema = metadata["schema"]
    data = metadata["data"]
    lines = [
        f"# {metadata['name']}",
        "",
        f"- **Task:** {metadata['task']}, predicting `{schema['target']}`",
        f"- **Model:** `{metadata['model']['type']}`",
        f"- **Data:** {data['rows']} rows ({data['train_rows']} train, "
        f"{data['test_rows']} test), {len(schema['features'])} features",
        f"- **Created:** {metadata['created_at']} with ml-pipeline-toolkit "
        f"{metadata['toolkit_version']}",
    ]
    if schema.get("positive_class") is not None:
        lines.append(f"- **Positive class:** `{schema['positive_class']}`")
    best = metadata["model"].get("best_params")
    if best:
        params = ", ".join(f"`{k}={v!r}`" for k, v in best.items())
        lines.append(f"- **Best parameters:** {params}")

    test = metadata["evaluation"]["metrics"]
    base = metadata["evaluation"].get("baseline") or {}
    lines += ["", "## Test set", ""]
    if base:
        lines += ["| Metric | Model | Baseline |", "| --- | ---: | ---: |"]
        for name, value in test.items():
            ref = base.get(name)
            lines.append(
                f"| {name} | {value:.4f} | {'' if ref is None else f'{ref:.4f}'} |"
            )
    else:
        lines += ["| Metric | Model |", "| --- | ---: |"]
        lines += [f"| {name} | {value:.4f} |" for name, value in test.items()]

    cv = metadata["evaluation"].get("cross_validation")
    if cv:
        lines += ["", "## Cross-validation (training set)", ""]
        lines += ["| Metric | Mean | Std |", "| --- | ---: | ---: |"]
        lines += [f"| {k} | {v['mean']:.4f} | {v['std']:.4f} |" for k, v in cv.items()]

    confusion = metadata["evaluation"].get("confusion_matrix")
    if confusion:
        labels = [str(label) for label in confusion["labels"]]
        lines += [
            "",
            "## Confusion matrix",
            "",
            "Rows are actual, columns predicted.",
            "",
        ]
        lines.append("| | " + " | ".join(labels) + " |")
        lines.append("| --- |" + " ---: |" * len(labels))
        for label, row in zip(labels, confusion["matrix"], strict=True):
            lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def is_better(metric: str, value: float, reference: float) -> bool:
    return value < reference if metric in LOWER_IS_BETTER else value > reference


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
