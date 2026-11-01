from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mlpipeline import DataError, Feature, Schema
from mlpipeline.config import DataConfig
from mlpipeline.data import load_table, prepare, split


def test_load_csv_and_jsonl(tmp_path: Path, churn_frame: pd.DataFrame) -> None:
    csv = tmp_path / "t.csv"
    churn_frame.to_csv(csv, index=False)
    assert load_table(str(csv)).shape == churn_frame.shape
    jsonl = tmp_path / "t.jsonl"
    churn_frame.to_json(jsonl, orient="records", lines=True)
    assert load_table(str(jsonl)).shape == churn_frame.shape


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("table.xlsx", "unsupported file type"),
        ("missing.csv", "file not found"),
        ("sklearn:mnist", "unknown dataset"),
    ],
)
def test_load_errors(source: str, message: str) -> None:
    with pytest.raises(DataError, match=message):
        load_table(source)


def test_load_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("a,b\n")
    with pytest.raises(DataError, match="no rows"):
        load_table(str(path))


def test_sklearn_dataset_has_named_labels() -> None:
    frame = load_table("sklearn:iris")
    assert set(frame["target"]) == {"setosa", "versicolor", "virginica"}


def test_prepare_infers_schema(churn_frame: pd.DataFrame) -> None:
    cfg = DataConfig(source="", target="churned", drop=("id",))
    X, y, schema = prepare(churn_frame, cfg, "classification")
    assert list(X.columns) == ["tenure", "charges", "plan", "region"]
    assert schema.of_kind("numeric") == ["tenure", "charges"]
    assert schema.of_kind("categorical") == ["plan", "region"]
    assert schema.classes == ("no", "yes")
    assert schema.positive_class == "yes"
    assert len(y) == len(churn_frame)


def test_prepare_overrides_kinds_and_selects_features(
    churn_frame: pd.DataFrame,
) -> None:
    cfg = DataConfig(
        source="",
        target="churned",
        features=("tenure", "plan"),
        categorical=("tenure",),
    )
    _, _, schema = prepare(churn_frame, cfg, "classification")
    assert schema.features == (
        Feature("tenure", "categorical"),
        Feature("plan", "categorical"),
    )


def test_prepare_drops_rows_without_target(churn_frame: pd.DataFrame) -> None:
    churn_frame.loc[:9, "churned"] = None
    X, _, _ = prepare(
        churn_frame, DataConfig(source="", target="churned"), "classification"
    )
    assert len(X) == len(churn_frame) - 10


@pytest.mark.parametrize(
    ("cfg", "task", "message"),
    [
        (DataConfig(source="", target="nope"), "classification", "target column"),
        (
            DataConfig(source="", target="churned", drop=("x",)),
            "classification",
            "not found: x",
        ),
        (DataConfig(source="", target="churned"), "regression", "must be numeric"),
        (
            DataConfig(source="", target="churned", positive_class="maybe"),
            "classification",
            "positive_class",
        ),
        (
            DataConfig(source="", target="churned", numeric=("plan",)),
            "classification",
            "non-numeric",
        ),
        (
            DataConfig(source="", target="churned", features=()),
            "classification",
            "no feature columns",
        ),
    ],
)
def test_prepare_errors(
    churn_frame: pd.DataFrame, cfg: DataConfig, task: str, message: str
) -> None:
    with pytest.raises(DataError, match=message):
        prepare(churn_frame, cfg, task)  # type: ignore[arg-type]


def test_single_class_is_rejected(churn_frame: pd.DataFrame) -> None:
    churn_frame["churned"] = "no"
    with pytest.raises(DataError, match="only one class"):
        prepare(churn_frame, DataConfig(source="", target="churned"), "classification")


def test_split_is_stratified_and_reproducible(churn_frame: pd.DataFrame) -> None:
    cfg = DataConfig(source="", target="churned", test_size=0.25)
    X, y, _ = prepare(churn_frame, cfg, "classification")
    _, X_test, _, y_test = split(X, y, cfg, "classification", seed=1)
    _, X_test_again, _, _ = split(X, y, cfg, "classification", seed=1)
    assert len(X_test) == 60
    assert X_test.index.equals(X_test_again.index)
    assert y_test.value_counts(normalize=True)["yes"] == pytest.approx(
        y.value_counts(normalize=True)["yes"], abs=0.02
    )


def test_split_falls_back_when_a_class_is_tiny(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = pd.DataFrame({"x": range(20), "y": ["a"] * 19 + ["b"]})
    cfg = DataConfig(source="", target="y")
    X, y, _ = prepare(frame, cfg, "classification")
    split(X, y, cfg, "classification", seed=0)
    assert "splitting randomly" in caplog.text


SCHEMA = Schema(
    target="y",
    task="classification",
    features=(Feature("age", "numeric"), Feature("plan", "categorical")),
    classes=("no", "yes"),
)


def test_check_records_reports_every_problem() -> None:
    errors = SCHEMA.check_records(
        [
            {"age": 30, "plan": "basic"},
            {"age": None, "plan": None},
            {"age": "thirty", "plan": "basic", "extra": 1},
            {"age": True, "plan": ["a"]},
            {"age": float("inf"), "plan": 3},
            {"plan": "basic"},
        ]
    )
    assert errors == [
        {"record": 2, "field": "extra", "error": "unknown field"},
        {"record": 2, "field": "age", "error": "expected a number or null"},
        {"record": 3, "field": "age", "error": "expected a number or null"},
        {
            "record": 3,
            "field": "plan",
            "error": "expected a string, number, boolean or null",
        },
        {"record": 4, "field": "age", "error": "expected a finite number"},
        {"record": 5, "field": "age", "error": "missing field"},
    ]


def test_check_frame() -> None:
    frame = pd.DataFrame({"plan": ["a"], "age": [1], "other": [0]})
    assert list(SCHEMA.check_frame(frame).columns) == ["age", "plan"]
    with pytest.raises(DataError, match="unknown column"):
        SCHEMA.check_frame(frame, strict=True)
    with pytest.raises(DataError, match="missing column"):
        SCHEMA.check_frame(frame[["plan"]])


def test_schema_round_trip() -> None:
    assert Schema.from_dict(SCHEMA.to_dict()) == SCHEMA


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
