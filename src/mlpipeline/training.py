"""Fitting the full pipeline, with optional search and cross-validation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from mlpipeline.config import PipelineConfig, Task
from mlpipeline.data import DataError, Schema
from mlpipeline.models import create_model
from mlpipeline.preprocessing import build_preprocessor

log = logging.getLogger(__name__)

DEFAULT_SCORING: dict[Task, str] = {
    "classification": "f1_macro",
    "regression": "neg_root_mean_squared_error",
}

# What cross-validation reports, by task: our name -> scikit-learn scorer.
CV_SCORERS: dict[Task, dict[str, str]] = {
    "classification": {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    },
    "regression": {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
}


@dataclass
class TrainResult:
    pipeline: Pipeline
    fit_seconds: float
    best_params: dict[str, Any] | None = None
    search_score: float | None = None
    search_scoring: str | None = None
    cross_validation: dict[str, dict[str, float]] | None = None


def build_pipeline(schema: Schema, config: PipelineConfig) -> Pipeline:
    """The unfitted pipeline: preprocessing then the configured model."""
    model = create_model(
        config.model.type, config.task, config.model.params, config.seed
    )
    return Pipeline(
        [
            ("preprocess", build_preprocessor(schema, config.preprocessing)),
            ("model", model),
        ]
    )


def train(
    X: pd.DataFrame, y: pd.Series, schema: Schema, config: PipelineConfig
) -> TrainResult:
    pipeline = build_pipeline(schema, config)
    result = TrainResult(pipeline=pipeline, fit_seconds=0.0)

    search = config.model.search
    if search is not None:
        scoring = search.scoring or DEFAULT_SCORING[config.task]
        grid = {_qualify(k): v for k, v in search.params.items()}
        cv = _splitter(config.task, y, search.cv, config.seed)
        if cv is None:
            raise DataError("too few rows per class for a hyperparameter search")
        searcher: GridSearchCV | RandomizedSearchCV
        if search.method == "grid":
            searcher = GridSearchCV(
                pipeline, grid, scoring=scoring, cv=cv, n_jobs=config.n_jobs, refit=True
            )
        else:
            searcher = RandomizedSearchCV(
                pipeline,
                grid,
                n_iter=search.n_iter,
                scoring=scoring,
                cv=cv,
                n_jobs=config.n_jobs,
                random_state=config.seed,
                refit=True,
            )
        log.info("searching %s over %s", search.method, ", ".join(search.params))
        started = time.perf_counter()
        searcher.fit(X, y)
        result.fit_seconds = time.perf_counter() - started
        result.pipeline = searcher.best_estimator_
        result.best_params = {
            k.removeprefix("model__"): _plain(v)
            for k, v in searcher.best_params_.items()
        }
        result.search_score = float(searcher.best_score_)
        result.search_scoring = scoring
        log.info(
            "best %s %.4f with %s", scoring, result.search_score, result.best_params
        )
    else:
        started = time.perf_counter()
        pipeline.fit(X, y)
        result.fit_seconds = time.perf_counter() - started

    if config.evaluation.cv_folds:
        result.cross_validation = cross_validation(
            result.pipeline, X, y, config.task, config.evaluation.cv_folds, config
        )
    log.info("fitted in %.2fs", result.fit_seconds)
    return result


def cross_validation(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task: Task,
    folds: int,
    config: PipelineConfig,
) -> dict[str, dict[str, float]] | None:
    """Mean and standard deviation of each metric across folds of the training set."""
    cv = _splitter(task, y, folds, config.seed)
    if cv is None:
        return None
    scorers = CV_SCORERS[task]
    scores = cross_validate(
        clone(pipeline), X, y, cv=cv, scoring=dict(scorers), n_jobs=config.n_jobs
    )
    out = {}
    for name, scorer in scorers.items():
        values = scores[f"test_{name}"]
        if scorer.startswith("neg_"):
            values = -values
        out[name] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return out


def _splitter(
    task: Task, y: pd.Series, folds: int, seed: int
) -> KFold | StratifiedKFold | None:
    if task == "regression":
        return KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=seed)
    smallest = int(y.value_counts().min())
    if smallest < 2:
        log.warning("a class has a single training row; skipping cross-validation")
        return None
    if smallest < folds:
        log.warning(
            "smallest class has %d rows; using %d folds, not %d",
            smallest,
            smallest,
            folds,
        )
        folds = smallest
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _qualify(param: str) -> str:
    """Search params name the model's own parameters unless they say otherwise."""
    if param.startswith(("model__", "preprocess__")):
        return param
    return f"model__{param}"


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
