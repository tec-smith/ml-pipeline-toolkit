"""Pipeline configuration: one YAML file describes a whole run.

The file is parsed into frozen dataclasses and validated up front, so a typo
in a key or an invalid choice fails before any data is loaded.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

Task = Literal["classification", "regression"]

TASKS = ("classification", "regression")
NUMERIC_IMPUTE = ("median", "mean", "zero")
NUMERIC_SCALE = ("standard", "minmax", "robust", "none")
CATEGORICAL_IMPUTE = ("constant", "most_frequent")
CATEGORICAL_ENCODE = ("onehot", "ordinal")
SEARCH_METHODS = ("grid", "random")

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ConfigError(ValueError):
    """A pipeline config is missing a value or has an invalid one."""


@dataclass(frozen=True)
class DataConfig:
    """Where the data comes from and which columns to use."""

    source: str
    target: str
    features: tuple[str, ...] | None = None
    drop: tuple[str, ...] = ()
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    test_size: float = 0.2
    stratify: bool = True
    positive_class: Any = None


@dataclass(frozen=True)
class NumericConfig:
    impute: str = "median"
    scale: str = "standard"


@dataclass(frozen=True)
class CategoricalConfig:
    impute: str = "constant"
    encode: str = "onehot"
    max_categories: int | None = 20
    min_frequency: int | float | None = None


@dataclass(frozen=True)
class PreprocessingConfig:
    numeric: NumericConfig = field(default_factory=NumericConfig)
    categorical: CategoricalConfig = field(default_factory=CategoricalConfig)


@dataclass(frozen=True)
class SearchConfig:
    """Hyperparameter search over the model's parameters."""

    params: dict[str, Any]
    method: str = "grid"
    n_iter: int = 20
    cv: int = 5
    scoring: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    search: SearchConfig | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    cv_folds: int = 5
    baseline: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    """A complete, validated pipeline run."""

    name: str
    task: Task
    data: DataConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output_dir: str = ""
    seed: int = 42
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if not self.output_dir:
            object.__setattr__(self, "output_dir", f"artifacts/{self.name}")

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: Sequence[str] = ()
    ) -> PipelineConfig:
        """Read a config file, applying ``key.path=value`` overrides first."""
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read {path}: {exc.strerror}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        return cls.from_dict(apply_overrides(raw, overrides))

    @classmethod
    def from_dict(cls, raw: Any) -> PipelineConfig:
        top = _section(
            raw,
            "config",
            {
                "name",
                "task",
                "seed",
                "n_jobs",
                "data",
                "preprocessing",
                "model",
                "evaluation",
                "output",
            },
        )
        name = _required(top, "name", "config", str)
        if not _NAME.match(name):
            raise ConfigError(
                "name: use letters, digits, '.', '_' and '-', starting with a "
                f"letter or digit, got {name!r}"
            )
        task = cast(Task, _choice(_required(top, "task", "config", str), "task", TASKS))
        seed = _typed(top.get("seed", 42), "seed", int)
        n_jobs = _typed(top.get("n_jobs"), "n_jobs", int, optional=True)
        output = _section(top.get("output"), "output", {"dir"})
        output_dir = _typed(output.get("dir", ""), "output.dir", str)
        return cls(
            name=name,
            task=task,
            data=_parse_data(top.get("data"), task),
            model=_parse_model(top.get("model")),
            preprocessing=_parse_preprocessing(top.get("preprocessing")),
            evaluation=_parse_evaluation(top.get("evaluation")),
            output_dir=output_dir,
            seed=seed,
            n_jobs=n_jobs,
        )

    def to_dict(self) -> dict[str, Any]:
        """The config as plain data that :meth:`from_dict` reads back."""
        out: dict[str, Any] = _plain(asdict(self))
        out["output"] = {"dir": out.pop("output_dir")}
        return out


def apply_overrides(raw: Any, overrides: Sequence[str]) -> Any:
    """Return a copy of ``raw`` with each ``a.b.c=value`` override applied.

    Values are parsed as YAML, so ``model.params.max_depth=null`` sets None
    and ``data.drop=[id]`` sets a list.
    """
    if not overrides:
        return raw
    if not isinstance(raw, dict):
        raise ConfigError("config: expected a mapping at the top level")
    out = copy.deepcopy(raw)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"override {item!r}: expected key.path=value")
        *parents, leaf = key.split(".")
        node = out
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {item!r}: {part} is not a mapping")
        node[leaf] = yaml.safe_load(value)
    return out


def _parse_data(raw: Any, task: Task) -> DataConfig:
    data = _section(
        raw,
        "data",
        {
            "source",
            "target",
            "features",
            "drop",
            "numeric",
            "categorical",
            "test_size",
            "stratify",
            "positive_class",
        },
        required=True,
    )
    features = data.get("features")
    test_size = _number(data.get("test_size", 0.2), "data.test_size")
    if not 0 < test_size < 1:
        raise ConfigError(f"data.test_size: must be between 0 and 1, got {test_size}")
    if task == "regression" and data.get("positive_class") is not None:
        raise ConfigError("data.positive_class: only applies to classification")
    cfg = DataConfig(
        source=_required(data, "source", "data", str),
        target=_required(data, "target", "data", str),
        features=None if features is None else _names(features, "data.features"),
        drop=_names(data.get("drop", []), "data.drop"),
        numeric=_names(data.get("numeric", []), "data.numeric"),
        categorical=_names(data.get("categorical", []), "data.categorical"),
        test_size=test_size,
        stratify=_typed(data.get("stratify", True), "data.stratify", bool),
        positive_class=data.get("positive_class"),
    )
    both = set(cfg.numeric) & set(cfg.categorical)
    if both:
        raise ConfigError(
            f"data: {', '.join(sorted(both))} listed as both numeric and categorical"
        )
    if cfg.target in (cfg.features or ()) or cfg.target in cfg.drop:
        raise ConfigError(
            f"data: target {cfg.target!r} can't also be a feature or dropped"
        )
    return cfg


def _parse_preprocessing(raw: Any) -> PreprocessingConfig:
    pre = _section(raw, "preprocessing", {"numeric", "categorical"})
    num = _section(pre.get("numeric"), "preprocessing.numeric", {"impute", "scale"})
    cat = _section(
        pre.get("categorical"),
        "preprocessing.categorical",
        {"impute", "encode", "max_categories", "min_frequency"},
    )
    where = "preprocessing.categorical"
    max_categories = _typed(
        cat.get("max_categories", 20), f"{where}.max_categories", int, optional=True
    )
    if max_categories is not None and max_categories < 2:
        raise ConfigError(f"{where}.max_categories: must be at least 2")
    min_frequency = cat.get("min_frequency")
    if min_frequency is not None:
        min_frequency = _number(min_frequency, f"{where}.min_frequency")
        if min_frequency <= 0:
            raise ConfigError(f"{where}.min_frequency: must be positive")
    return PreprocessingConfig(
        numeric=NumericConfig(
            impute=_choice(
                num.get("impute", "median"),
                "preprocessing.numeric.impute",
                NUMERIC_IMPUTE,
            ),
            scale=_choice(
                num.get("scale", "standard"),
                "preprocessing.numeric.scale",
                NUMERIC_SCALE,
            ),
        ),
        categorical=CategoricalConfig(
            impute=_choice(
                cat.get("impute", "constant"), f"{where}.impute", CATEGORICAL_IMPUTE
            ),
            encode=_choice(
                cat.get("encode", "onehot"), f"{where}.encode", CATEGORICAL_ENCODE
            ),
            max_categories=max_categories,
            min_frequency=min_frequency,
        ),
    )


def _parse_model(raw: Any) -> ModelConfig:
    model = _section(raw, "model", {"type", "params", "search"}, required=True)
    params = _section(model.get("params"), "model.params", None)
    search = None
    if model.get("search") is not None:
        s = _section(
            model["search"],
            "model.search",
            {"method", "params", "n_iter", "cv", "scoring"},
        )
        grid = _section(s.get("params"), "model.search.params", None, required=True)
        if not grid:
            raise ConfigError("model.search.params: list at least one parameter")
        for key, values in grid.items():
            if not isinstance(values, list) or not values:
                raise ConfigError(
                    f"model.search.params.{key}: expected a non-empty list of values"
                )
        cv = _typed(s.get("cv", 5), "model.search.cv", int)
        if cv < 2:
            raise ConfigError("model.search.cv: must be at least 2")
        n_iter = _typed(s.get("n_iter", 20), "model.search.n_iter", int)
        if n_iter < 1:
            raise ConfigError("model.search.n_iter: must be at least 1")
        search = SearchConfig(
            params=dict(grid),
            method=_choice(
                s.get("method", "grid"), "model.search.method", SEARCH_METHODS
            ),
            n_iter=n_iter,
            cv=cv,
            scoring=_typed(
                s.get("scoring"), "model.search.scoring", str, optional=True
            ),
        )
    return ModelConfig(
        type=_required(model, "type", "model", str),
        params=dict(params),
        search=search,
    )


def _parse_evaluation(raw: Any) -> EvaluationConfig:
    ev = _section(raw, "evaluation", {"cv_folds", "baseline"})
    folds = _typed(ev.get("cv_folds", 5), "evaluation.cv_folds", int)
    if folds == 1 or folds < 0:
        raise ConfigError("evaluation.cv_folds: use 0 to turn it off, or 2 or more")
    return EvaluationConfig(
        cv_folds=folds,
        baseline=_typed(ev.get("baseline", True), "evaluation.baseline", bool),
    )


def _section(
    raw: Any, where: str, allowed: set[str] | None, required: bool = False
) -> dict[str, Any]:
    if raw is None:
        if required:
            raise ConfigError(f"{where}: missing")
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    if allowed is not None:
        unknown = sorted(str(k) for k in raw if k not in allowed)
        if unknown:
            raise ConfigError(
                f"{where}: unknown key {', '.join(unknown)} "
                f"(expected {', '.join(sorted(allowed))})"
            )
    return dict(raw)


def _required(section: dict[str, Any], key: str, where: str, kind: type) -> Any:
    if section.get(key) is None:
        raise ConfigError(
            f"{where}.{key}: missing" if where != "config" else f"{key}: missing"
        )
    return _typed(section[key], key if where == "config" else f"{where}.{key}", kind)


def _typed(value: Any, where: str, kind: type, optional: bool = False) -> Any:
    if value is None and optional:
        return None
    # bool is a subclass of int, but "seed: true" is a mistake, not a seed.
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise ConfigError(
            f"{where}: expected {kind.__name__}, got {type(value).__name__}"
        )
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where}: expected a number, got {type(value).__name__}")
    return value


def _choice(value: Any, where: str, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ConfigError(
            f"{where}: expected one of {', '.join(choices)}, got {value!r}"
        )
    return cast(str, value)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _names(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ConfigError(f"{where}: expected a list of column names")
    return tuple(value)


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
