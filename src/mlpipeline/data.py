"""Loading tables, inferring their schema and checking new data against it.

The :class:`Schema` is the contract between training and serving: it records
each input column and whether it is numeric or categorical, and is saved in
the model bundle so prediction requests can be validated before they reach
the model.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlpipeline.config import DataConfig, Task

log = logging.getLogger(__name__)

FeatureKind = Literal["numeric", "categorical"]

# Datasets that ship inside scikit-learn, so examples run without a download.
SKLEARN_DATASETS = ("breast_cancer", "diabetes", "iris", "wine")
_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".parquet": pd.read_parquet,
    ".jsonl": lambda p: pd.read_json(p, lines=True),
    ".json": pd.read_json,
}


class DataError(ValueError):
    """The data can't be used: a missing column, a bad value, too few rows."""


@dataclass(frozen=True)
class Feature:
    name: str
    kind: FeatureKind


@dataclass(frozen=True)
class Schema:
    """The input columns a model expects, and what it predicts."""

    target: str
    task: Task
    features: tuple[Feature, ...]
    classes: tuple[Any, ...] | None = None
    positive_class: Any = None
    _by_name: dict[str, Feature] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {f.name: f for f in self.features})

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]

    def of_kind(self, kind: FeatureKind) -> list[str]:
        return [f.name for f in self.features if f.kind == kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "task": self.task,
            "features": [{"name": f.name, "kind": f.kind} for f in self.features],
            "classes": None if self.classes is None else list(self.classes),
            "positive_class": self.positive_class,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Schema:
        classes = raw.get("classes")
        return cls(
            target=raw["target"],
            task=raw["task"],
            features=tuple(Feature(f["name"], f["kind"]) for f in raw["features"]),
            classes=None if classes is None else tuple(classes),
            positive_class=raw.get("positive_class"),
        )

    def check_frame(self, frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """Return the feature columns of ``frame`` in training order.

        Missing columns are an error. Extra columns are ignored, or an error
        if ``strict`` is set.
        """
        missing = [n for n in self.names if n not in frame.columns]
        if missing:
            raise DataError(f"missing column(s): {', '.join(missing)}")
        if strict:
            extra = [str(c) for c in frame.columns if c not in self._by_name]
            if extra:
                raise DataError(f"unknown column(s): {', '.join(extra)}")
        return frame[self.names]

    def check_records(
        self, records: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate JSON-style records and list every problem found.

        Each problem is ``{"record": index, "field": name, "error": message}``.
        Every feature must be present (``null`` is allowed and imputed), and
        unknown fields are rejected so a misspelt name doesn't silently become
        a missing value.
        """
        errors: list[dict[str, Any]] = []
        for i, record in enumerate(records):
            for name in record:
                if name not in self._by_name:
                    errors.append(
                        {"record": i, "field": name, "error": "unknown field"}
                    )
            for feature in self.features:
                if feature.name not in record:
                    errors.append(
                        {"record": i, "field": feature.name, "error": "missing field"}
                    )
                    continue
                problem = _check_value(record[feature.name], feature.kind)
                if problem:
                    errors.append(
                        {"record": i, "field": feature.name, "error": problem}
                    )
        return errors


def load_table(source: str) -> pd.DataFrame:
    """Load a CSV, TSV, Parquet or JSON file, or ``sklearn:<dataset>``."""
    if source.startswith("sklearn:"):
        return _load_sklearn(source.removeprefix("sklearn:"))
    path = Path(source)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise DataError(
            f"{source}: unsupported file type; use one of "
            f"{', '.join(sorted(_READERS))} or sklearn:<dataset>"
        )
    if not path.is_file():
        raise DataError(f"{source}: file not found")
    try:
        frame = reader(path)
    except ImportError as exc:
        raise DataError(
            f"{source}: reading this format needs an extra package "
            "(pip install 'ml-pipeline-toolkit[parquet]')"
        ) from exc
    except (ValueError, pd.errors.ParserError) as exc:
        raise DataError(f"{source}: can't parse: {exc}") from exc
    if frame.empty:
        raise DataError(f"{source}: no rows")
    return frame


def _load_sklearn(name: str) -> pd.DataFrame:
    from sklearn import datasets

    if name not in SKLEARN_DATASETS:
        raise DataError(
            f"sklearn:{name}: unknown dataset; use one of {', '.join(SKLEARN_DATASETS)}"
        )
    bunch = getattr(datasets, f"load_{name}")(as_frame=True)
    frame: pd.DataFrame = bunch.frame
    # Named labels read better in reports and API responses than 0, 1, 2.
    if name in ("iris", "wine", "breast_cancer"):
        frame["target"] = np.asarray(bunch.target_names)[bunch.target]
    return frame


def prepare(
    frame: pd.DataFrame, cfg: DataConfig, task: Task
) -> tuple[pd.DataFrame, pd.Series, Schema]:
    """Split a table into features and target and infer the schema."""
    if cfg.target not in frame.columns:
        raise DataError(f"target column {cfg.target!r} not found")
    unknown = [
        c
        for c in (*(cfg.features or ()), *cfg.drop, *cfg.numeric, *cfg.categorical)
        if c not in frame.columns
    ]
    if unknown:
        raise DataError(f"column(s) not found: {', '.join(dict.fromkeys(unknown))}")

    missing_target = frame[cfg.target].isna()
    if missing_target.any():
        log.warning(
            "dropping %d row(s) with no target value", int(missing_target.sum())
        )
        frame = frame.loc[~missing_target]

    if cfg.features is not None:
        columns = list(cfg.features)
    else:
        skip = {cfg.target, *cfg.drop}
        columns = [c for c in frame.columns if c not in skip]
    if not columns:
        raise DataError("no feature columns left after dropping")

    X = frame[columns].copy()
    y = frame[cfg.target]
    features = tuple(Feature(str(c), _infer_kind(X[c], str(c), cfg)) for c in columns)

    classes = None
    positive = None
    if task == "classification":
        classes = tuple(_plain(v) for v in sorted(y.unique()))
        if len(classes) < 2:
            raise DataError(f"target {cfg.target!r} has only one class")
        if len(classes) == 2:
            positive = classes[-1] if cfg.positive_class is None else cfg.positive_class
            if positive not in classes:
                raise DataError(
                    f"data.positive_class {positive!r} is not one of {list(classes)}"
                )
    else:
        if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
            raise DataError(f"target {cfg.target!r} must be numeric for regression")

    schema = Schema(cfg.target, task, features, classes, positive)
    log.info(
        "%d rows, %d numeric and %d categorical feature(s)",
        len(X),
        len(schema.of_kind("numeric")),
        len(schema.of_kind("categorical")),
    )
    return X, y, schema


def split(
    X: pd.DataFrame, y: pd.Series, cfg: DataConfig, task: Task, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """A reproducible train/test split, stratified where it can be."""
    stratify = None
    if task == "classification" and cfg.stratify:
        smallest = int(y.value_counts().min())
        n_test = math.ceil(len(y) * cfg.test_size)
        if smallest >= 2 and n_test >= y.nunique():
            stratify = y
        else:
            log.warning(
                "too few rows per class to stratify the split; splitting randomly"
            )
    if len(y) < 4:
        raise DataError(f"need at least 4 rows to split, got {len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=seed, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def _infer_kind(column: pd.Series, name: str, cfg: DataConfig) -> FeatureKind:
    if name in cfg.categorical:
        return "categorical"
    if name in cfg.numeric:
        if not pd.api.types.is_numeric_dtype(column):
            converted = pd.to_numeric(column, errors="coerce")
            bad = converted.isna() & column.notna()
            if bad.any():
                raise DataError(
                    f"column {name!r} is listed as numeric but has non-numeric "
                    f"values such as {column[bad].iloc[0]!r}"
                )
        return "numeric"
    if pd.api.types.is_bool_dtype(column) or not pd.api.types.is_numeric_dtype(column):
        return "categorical"
    return "numeric"


def _check_value(value: Any, kind: FeatureKind) -> str | None:
    if value is None:
        return None
    if kind == "numeric":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return "expected a number or null"
        if isinstance(value, float) and not math.isfinite(value):
            return "expected a finite number"
        return None
    if not isinstance(value, str | int | float | bool):
        return "expected a string, number, boolean or null"
    return None


def _plain(value: Any) -> Any:
    """numpy scalars to Python ones, so they survive a round trip to JSON."""
    return value.item() if isinstance(value, np.generic) else value


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
