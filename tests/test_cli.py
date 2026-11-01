from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from mlpipeline.cli import main


@pytest.fixture
def config_file(tmp_path: Path, config_dict: dict[str, Any]) -> Path:
    path = tmp_path / "churn.yml"
    path.write_text(yaml.safe_dump(config_dict))
    return path


def test_run_info_predict_evaluate(
    config_file: Path,
    churn_csv: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "cli-bundle"
    assert (
        main(
            [
                "run",
                str(config_file),
                "-o",
                str(bundle),
                "--set",
                "evaluation.cv_folds=0",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "roc_auc" in out
    assert "better" in out

    assert main(["info", str(bundle)]) == 0
    assert "## Test set" in capsys.readouterr().out
    assert main(["info", str(bundle), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["config"]["evaluation"]["cv_folds"] == 0

    predictions = tmp_path / "out" / "predictions.csv"
    assert (
        main(
            ["predict", str(bundle), str(churn_csv), "-o", str(predictions), "--proba"]
        )
        == 0
    )
    frame = pd.read_csv(predictions)
    assert {"id", "churned_predicted", "churned_predicted_proba_yes"} <= set(
        frame.columns
    )

    assert main(["evaluate", str(bundle), str(churn_csv), "--json"]) == 0
    assert "accuracy" in json.loads(capsys.readouterr().out)["metrics"]


def test_run_json(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(config_file), "--json"]) == 0
    assert "baseline" in json.loads(capsys.readouterr().out)


def test_check(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "plan (categorical)" in out
    assert "classes: no, yes" in out
    assert (
        main(
            ["check", str(config_file), "--no-data", "--set", "data.source=missing.csv"]
        )
        == 0
    )


def test_exit_codes(
    config_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert main(["check", str(config_file), "--set", "task=clustering"]) == 2
    assert "invalid config" in caplog.text
    assert main(["check", str(config_file), "--set", "model.type=nope"]) == 2
    assert main(["check", str(config_file), "--set", "data.target=nope"]) == 1
    assert main(["info", str(tmp_path)]) == 1


def test_models(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models", "--task", "regression"]) == 0
    out = capsys.readouterr().out
    assert "ridge" in out
    assert "classification" not in out


def test_serve_needs_a_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_DIR", raising=False)
    assert main(["serve"]) == 2


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
