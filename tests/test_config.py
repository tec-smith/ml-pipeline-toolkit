from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mlpipeline import ConfigError, PipelineConfig
from mlpipeline.config import apply_overrides


def minimal(**changes: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "demo",
        "task": "classification",
        "data": {"source": "sklearn:iris", "target": "target"},
        "model": {"type": "logistic_regression"},
    }
    raw.update(changes)
    return raw


def test_defaults() -> None:
    cfg = PipelineConfig.from_dict(minimal())
    assert cfg.output_dir == "artifacts/demo"
    assert cfg.seed == 42
    assert cfg.data.test_size == 0.2
    assert cfg.preprocessing.numeric.scale == "standard"
    assert cfg.preprocessing.categorical.encode == "onehot"
    assert cfg.evaluation.cv_folds == 5
    assert cfg.model.search is None


def test_full_config_round_trips_to_dict() -> None:
    raw = minimal(
        seed=7,
        n_jobs=2,
        preprocessing={"numeric": {"impute": "mean", "scale": "robust"}},
        model={
            "type": "random_forest",
            "params": {"n_estimators": 10},
            "search": {
                "method": "random",
                "n_iter": 3,
                "params": {"max_depth": [2, None]},
            },
        },
        output={"dir": "out/demo"},
    )
    cfg = PipelineConfig.from_dict(raw)
    assert cfg.model.search is not None
    assert cfg.model.search.method == "random"
    out = cfg.to_dict()
    assert out["output"] == {"dir": "out/demo"}
    assert out["model"]["params"] == {"n_estimators": 10}
    assert PipelineConfig.from_dict(out) == cfg


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "clustering"}, "task: expected one of"),
        ({"name": "../escape"}, "name: use letters"),
        ({"surprise": 1}, "config: unknown key surprise"),
        ({"seed": True}, "seed: expected int"),
        ({"data": {"source": "x.csv"}}, "data.target: missing"),
        (
            {"data": {"source": "x.csv", "target": "y", "test_size": 1.5}},
            "between 0 and 1",
        ),
        (
            {"data": {"source": "x.csv", "target": "y", "drop": "id"}},
            "list of column names",
        ),
        ({"data": {"source": "x.csv", "target": "y", "drop": ["y"]}}, "can't also be"),
        (
            {
                "data": {
                    "source": "x.csv",
                    "target": "y",
                    "numeric": ["a"],
                    "categorical": ["a"],
                }
            },
            "both numeric and categorical",
        ),
        ({"model": {}}, "model.type: missing"),
        (
            {"model": {"type": "svm", "search": {"params": {}}}},
            "at least one parameter",
        ),
        ({"model": {"type": "svm", "search": {"params": {"C": 1}}}}, "non-empty list"),
        (
            {"model": {"type": "svm", "search": {"params": {"C": [1]}, "cv": 1}}},
            "at least 2",
        ),
        (
            {"preprocessing": {"numeric": {"scale": "log"}}},
            "preprocessing.numeric.scale",
        ),
        ({"preprocessing": {"categorical": {"max_categories": 1}}}, "at least 2"),
        ({"evaluation": {"cv_folds": 1}}, "use 0 to turn it off"),
    ],
)
def test_invalid_configs(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        PipelineConfig.from_dict(minimal(**changes))


def test_positive_class_only_for_classification() -> None:
    raw = minimal(task="regression")
    raw["data"]["positive_class"] = "yes"
    with pytest.raises(ConfigError, match="only applies to classification"):
        PipelineConfig.from_dict(raw)


def test_from_yaml_with_overrides(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text(
        "name: demo\ntask: regression\n"
        "data: {source: 'sklearn:diabetes', target: target}\n"
        "model: {type: ridge}\n"
    )
    cfg = PipelineConfig.from_yaml(
        path, ["model.params.alpha=0.5", "evaluation.cv_folds=0", "data.drop=[age]"]
    )
    assert cfg.model.params == {"alpha": 0.5}
    assert cfg.evaluation.cv_folds == 0
    assert cfg.data.drop == ("age",)


def test_from_yaml_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        PipelineConfig.from_yaml(tmp_path / "missing.yml")
    bad = tmp_path / "bad.yml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        PipelineConfig.from_yaml(bad)


def test_override_errors() -> None:
    with pytest.raises(ConfigError, match=r"expected key\.path=value"):
        apply_overrides({}, ["nonsense"])
    with pytest.raises(ConfigError, match="is not a mapping"):
        apply_overrides({"name": "x"}, ["name.inner=1"])


def test_examples_are_valid() -> None:
    for path in sorted(Path("examples").glob("*.yml")):
        PipelineConfig.from_yaml(path)


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
