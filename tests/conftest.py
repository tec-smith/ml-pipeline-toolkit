from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlpipeline import PipelineConfig


@pytest.fixture
def churn_frame() -> pd.DataFrame:
    """A small mixed-type table where the label depends on the features."""
    rng = np.random.default_rng(0)
    n = 240
    tenure = rng.integers(1, 60, n)
    plan = rng.choice(["basic", "plus", "pro"], n)
    region = rng.choice(["north", "south"], n)
    charges = rng.normal(50, 10, n).round(2)
    score = -0.08 * tenure + 1.5 * (plan == "basic") + rng.normal(0, 0.5, n)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "tenure": tenure,
            "charges": charges,
            "plan": plan,
            "region": region,
            "churned": np.where(score > np.median(score), "yes", "no"),
        }
    )
    frame.loc[::17, "charges"] = np.nan
    frame.loc[::23, "region"] = np.nan
    return frame


@pytest.fixture
def churn_csv(tmp_path: Path, churn_frame: pd.DataFrame) -> Path:
    path = tmp_path / "churn.csv"
    churn_frame.to_csv(path, index=False)
    return path


def churn_config(source: Path | str, output: Path, **changes: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "churn",
        "task": "classification",
        "data": {"source": str(source), "target": "churned", "drop": ["id"]},
        "model": {"type": "logistic_regression"},
        "evaluation": {"cv_folds": 3},
        "output": {"dir": str(output)},
    }
    for key, value in changes.items():
        config[key] = value
    return config


@pytest.fixture
def config_dict(churn_csv: Path, tmp_path: Path) -> dict[str, Any]:
    return churn_config(churn_csv, tmp_path / "bundle")


@pytest.fixture
def config(config_dict: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig.from_dict(config_dict)


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
