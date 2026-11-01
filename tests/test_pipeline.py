from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlpipeline import (
    BundleError,
    PipelineConfig,
    evaluate_bundle,
    load_bundle,
    run_pipeline,
)
from mlpipeline.data import Schema
from mlpipeline.evaluation import (
    classification_report,
    regression_metrics,
    render_report,
)


def test_run_saves_a_complete_bundle(
    config: PipelineConfig, churn_frame: pd.DataFrame
) -> None:
    result = run_pipeline(config)
    files = {p.name for p in result.output_dir.iterdir()}
    assert files == {"model.joblib", "metadata.json", "report.md"}

    meta = json.loads((result.output_dir / "metadata.json").read_text())
    assert meta["bundle_format"] == 1
    assert meta["schema"]["classes"] == ["no", "yes"]
    assert meta["data"] == {
        "source": config.data.source,
        "rows": 240,
        "train_rows": 192,
        "test_rows": 48,
    }
    assert set(meta["evaluation"]["cross_validation"]) == {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
    }
    assert len(meta["model_sha256"]) == 64
    assert result.metrics["roc_auc"] > result.baseline["roc_auc"]  # type: ignore[index]

    bundle = load_bundle(result.output_dir)
    rows = churn_frame.head(5)
    assert (bundle.predict(rows) == result.bundle.predict(rows)).all()
    proba = bundle.predict_proba(rows)
    assert list(proba.columns) == ["no", "yes"]
    assert np.allclose(proba.sum(axis=1), 1)


def test_run_is_reproducible(config: PipelineConfig, tmp_path: Path) -> None:
    first = run_pipeline(config, tmp_path / "a")
    second = run_pipeline(config, tmp_path / "b")
    assert first.metrics == second.metrics


def test_run_with_search_and_dataframe(
    config_dict: dict[str, Any], churn_frame: pd.DataFrame
) -> None:
    config_dict["model"] = {
        "type": "random_forest",
        "params": {"n_estimators": 10},
        "search": {"params": {"max_depth": [2, 4]}, "cv": 3},
    }
    config_dict["evaluation"] = {"cv_folds": 0, "baseline": False}
    result = run_pipeline(PipelineConfig.from_dict(config_dict), frame=churn_frame)
    meta = result.bundle.metadata
    assert meta["model"]["best_params"]["max_depth"] in (2, 4)
    assert meta["model"]["search"]["scoring"] == "f1_macro"
    assert meta["data"]["source"] == "<dataframe>"
    assert meta["evaluation"]["cross_validation"] is None
    assert result.baseline is None
    assert "Best parameters" in result.report


def test_regression_run(tmp_path: Path) -> None:
    config = PipelineConfig.from_dict(
        {
            "name": "diabetes",
            "task": "regression",
            "data": {"source": "sklearn:diabetes", "target": "target"},
            "model": {"type": "ridge"},
            "evaluation": {"cv_folds": 3},
        }
    )
    result = run_pipeline(config, tmp_path / "diabetes")
    assert result.metrics["r2"] > 0.3
    assert result.metrics["rmse"] < result.baseline["rmse"]  # type: ignore[index]
    assert not result.bundle.can_predict_proba
    assert "Confusion matrix" not in result.report


def test_multiclass_run(tmp_path: Path) -> None:
    config = PipelineConfig.from_dict(
        {
            "name": "iris",
            "task": "classification",
            "data": {"source": "sklearn:iris", "target": "target"},
            "model": {"type": "logistic_regression"},
        }
    )
    result = run_pipeline(config, tmp_path / "iris")
    assert result.metrics["accuracy"] > 0.85
    assert result.bundle.schema.positive_class is None
    assert result.bundle.metadata["evaluation"]["confusion_matrix"]["labels"] == [
        "setosa",
        "versicolor",
        "virginica",
    ]


def test_evaluate_bundle_on_new_data(
    config: PipelineConfig, churn_frame: pd.DataFrame
) -> None:
    bundle = run_pipeline(config).bundle
    scores = evaluate_bundle(bundle, churn_frame)
    assert 0.5 < scores["metrics"]["accuracy"] <= 1
    with pytest.raises(ValueError, match="target column"):
        evaluate_bundle(bundle, churn_frame.drop(columns="churned"))


def test_tampered_bundle_is_refused(config: PipelineConfig) -> None:
    out = run_pipeline(config).output_dir
    with (out / "model.joblib").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(BundleError, match="checksum mismatch"):
        load_bundle(out)


def test_bundle_errors(tmp_path: Path, config: PipelineConfig) -> None:
    with pytest.raises(BundleError, match="not a model bundle"):
        load_bundle(tmp_path)
    out = run_pipeline(config).output_dir
    meta = json.loads((out / "metadata.json").read_text())
    (out / "metadata.json").write_text(json.dumps({**meta, "bundle_format": 99}))
    with pytest.raises(BundleError, match="format 99"):
        load_bundle(out)


def test_save_replaces_a_bundle_but_not_other_directories(
    config: PipelineConfig, tmp_path: Path
) -> None:
    out = tmp_path / "bundle"
    run_pipeline(config, out)
    run_pipeline(config, out)
    assert not list(tmp_path.glob(".bundle*"))

    other = tmp_path / "precious"
    other.mkdir()
    (other / "notes.txt").write_text("keep me")
    with pytest.raises(BundleError, match="isn't a model bundle"):
        run_pipeline(config, other)
    assert (other / "notes.txt").read_text() == "keep me"


def test_version_mismatch_is_logged(
    config: PipelineConfig, caplog: pytest.LogCaptureFixture
) -> None:
    out = run_pipeline(config).output_dir
    meta = json.loads((out / "metadata.json").read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    (out / "metadata.json").write_text(json.dumps(meta))
    load_bundle(out)
    assert "trained with scikit-learn 0.0.1" in caplog.text


def test_metrics_known_values() -> None:
    schema = Schema("y", "classification", (), ("neg", "pos"), "pos")
    report = classification_report(
        ["pos", "pos", "neg", "neg"],
        ["pos", "neg", "neg", "neg"],
        np.array([[0.2, 0.8], [0.6, 0.4], [0.9, 0.1], [0.7, 0.3]]),
        schema,
    )
    metrics = report["metrics"]
    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["roc_auc"] == 1.0
    assert report["confusion_matrix"]["matrix"] == [[2, 0], [1, 1]]

    reg = regression_metrics([1, 2, 3], [1, 2, 5])
    assert reg["mae"] == pytest.approx(2 / 3)
    assert reg["rmse"] == pytest.approx((4 / 3) ** 0.5)
    assert "mape" not in regression_metrics([0, 1], [0, 1])


def test_report_without_baseline() -> None:
    meta = {
        "name": "m",
        "task": "regression",
        "created_at": "now",
        "toolkit_version": "0",
        "schema": {"target": "y", "features": [], "positive_class": None},
        "data": {"rows": 1, "train_rows": 1, "test_rows": 0},
        "model": {"type": "ridge"},
        "evaluation": {"metrics": {"r2": 0.5}},
    }
    assert "| r2 | 0.5000 |" in render_report(meta)


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
