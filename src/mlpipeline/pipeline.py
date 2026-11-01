"""A whole run, from config to saved bundle."""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd

from mlpipeline import data, evaluation, training
from mlpipeline.artifacts import ModelBundle, save_bundle
from mlpipeline.config import PipelineConfig
from mlpipeline.data import DataError

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    bundle: ModelBundle
    output_dir: Path
    report: str
    seconds: float

    @property
    def metrics(self) -> dict[str, float]:
        result: dict[str, float] = self.bundle.metadata["evaluation"]["metrics"]
        return result

    @property
    def baseline(self) -> dict[str, float] | None:
        result: dict[str, float] | None = self.bundle.metadata["evaluation"].get(
            "baseline"
        )
        return result


def run_pipeline(
    config: PipelineConfig,
    output_dir: str | Path | None = None,
    frame: pd.DataFrame | None = None,
) -> RunResult:
    """Load, prepare, split, train, evaluate and save.

    Pass ``frame`` to use a DataFrame you already have instead of reading
    ``data.source``.
    """
    started = time.perf_counter()
    target_dir = Path(output_dir or config.output_dir)
    log.info("run %r: %s with %s", config.name, config.task, config.model.type)

    table = data.load_table(config.data.source) if frame is None else frame
    X, y, schema = data.prepare(table, config.data, config.task)
    X_train, X_test, y_train, y_test = data.split(
        X, y, config.data, config.task, config.seed
    )

    trained = training.train(X_train, y_train, schema, config)
    scores = evaluation.evaluate(trained.pipeline, X_test, y_test, schema)
    if config.evaluation.baseline:
        scores["baseline"] = evaluation.baseline(
            X_train, y_train, X_test, y_test, schema
        )
    scores["cross_validation"] = trained.cross_validation

    metadata: dict[str, Any] = {
        "name": config.name,
        "task": config.task,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "toolkit_version": _version("ml-pipeline-toolkit"),
        "versions": {
            "python": platform.python_version(),
            **{p: _version(p) for p in ("scikit-learn", "pandas", "numpy")},
        },
        "schema": schema.to_dict(),
        "data": {
            "source": config.data.source if frame is None else "<dataframe>",
            "rows": len(X),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        "model": {
            "type": config.model.type,
            "params": config.model.params,
            "best_params": trained.best_params,
            "search": None
            if trained.search_score is None
            else {
                "scoring": trained.search_scoring,
                "best_score": trained.search_score,
            },
            "fit_seconds": round(trained.fit_seconds, 3),
        },
        "evaluation": scores,
        "config": config.to_dict(),
    }
    bundle = ModelBundle(pipeline=trained.pipeline, metadata=metadata)
    report = evaluation.render_report(metadata)
    save_bundle(bundle, target_dir, report)
    return RunResult(bundle, target_dir, report, time.perf_counter() - started)


def evaluate_bundle(bundle: ModelBundle, frame: pd.DataFrame) -> dict[str, Any]:
    """Score a saved model on new labelled data, such as last month's outcomes."""
    schema = bundle.schema
    if schema.target not in frame.columns:
        raise DataError(f"target column {schema.target!r} not found")
    frame = frame.loc[frame[schema.target].notna()]
    if frame.empty:
        raise DataError("no rows with a target value")
    X = schema.check_frame(frame)
    return evaluation.evaluate(bundle.pipeline, X, frame[schema.target], schema)


def _version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
