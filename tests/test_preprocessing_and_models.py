from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

from mlpipeline import (
    ConfigError,
    DataError,
    Feature,
    Schema,
    available_models,
    create_model,
    register_model,
)
from mlpipeline.config import CategoricalConfig, NumericConfig, PreprocessingConfig
from mlpipeline.preprocessing import SchemaCoercer, build_preprocessor

SCHEMA = Schema(
    target="y",
    task="classification",
    features=(
        Feature("age", "numeric"),
        Feature("code", "categorical"),
        Feature("plan", "categorical"),
    ),
)
TRAIN = pd.DataFrame(
    {
        "age": [20, 30, None, 50],
        "code": [1.0, 2.0, 1.0, None],
        "plan": ["a", "b", "a", "c"],
    }
)


def test_coercer_makes_categories_consistent() -> None:
    coercer = SchemaCoercer(["age"], ["code", "plan"])
    out = coercer.transform(pd.DataFrame({"age": ["7"], "code": [1], "plan": [True]}))
    assert out["age"].dtype == np.float64
    assert out.loc[0, "code"] == "1"
    assert out.loc[0, "plan"] == "True"
    floats = coercer.transform(TRAIN)
    assert floats["code"].tolist()[:3] == ["1", "2", "1"]
    assert pd.isna(floats["code"].iloc[3])


def test_coercer_rejects_non_numeric_values() -> None:
    with pytest.raises(DataError, match="'age': expected numbers"):
        SchemaCoercer(["age"], []).transform(pd.DataFrame({"age": ["old"]}))


def test_preprocessor_imputes_encodes_and_handles_unknowns() -> None:
    pre = build_preprocessor(SCHEMA, PreprocessingConfig())
    out = pre.fit_transform(TRAIN)
    assert not np.isnan(out).any()
    # 1 numeric + codes {1, 2, missing} + plans {a, b, c}
    assert out.shape == (4, 7)
    unseen = pre.transform(pd.DataFrame({"age": [40], "code": [9], "plan": ["z"]}))
    assert unseen.shape == (1, 7)


@pytest.mark.parametrize("scale", ["standard", "minmax", "robust", "none"])
@pytest.mark.parametrize("impute", ["median", "mean", "zero"])
def test_numeric_options(scale: str, impute: str) -> None:
    cfg = PreprocessingConfig(numeric=NumericConfig(impute=impute, scale=scale))
    out = build_preprocessor(SCHEMA, cfg).fit_transform(TRAIN)
    assert not np.isnan(out).any()


def test_ordinal_encoding_and_most_frequent() -> None:
    cfg = PreprocessingConfig(
        categorical=CategoricalConfig(impute="most_frequent", encode="ordinal")
    )
    pre = build_preprocessor(SCHEMA, cfg)
    assert pre.fit_transform(TRAIN).shape == (4, 3)
    assert (
        pre.transform(pd.DataFrame({"age": [1], "code": [5], "plan": ["z"]}))[0, 2]
        == -1
    )


def test_rare_categories_are_grouped() -> None:
    cfg = PreprocessingConfig(
        categorical=CategoricalConfig(max_categories=None, min_frequency=2)
    )
    out = build_preprocessor(SCHEMA, cfg).fit_transform(TRAIN)
    # code: "1" and infrequent; plan: "a" and infrequent
    assert out.shape == (4, 5)


def test_create_model_seeds_random_state() -> None:
    model = create_model("random_forest", "classification", {"n_estimators": 5}, seed=3)
    assert isinstance(model, RandomForestClassifier)
    assert model.get_params()["random_state"] == 3
    explicit = create_model(
        "random_forest", "classification", {"random_state": 9}, seed=3
    )
    assert explicit.get_params()["random_state"] == 9


def test_create_model_errors() -> None:
    with pytest.raises(ConfigError, match="is a regression model"):
        create_model("ridge", "classification", {})
    with pytest.raises(ConfigError, match=r"model\.params"):
        create_model("ridge", "regression", {"bogus": 1})
    with pytest.raises(ConfigError, match="can't import"):
        create_model("nowhere.module:Thing", "regression", {})


def test_import_path_and_registration() -> None:
    assert isinstance(
        create_model("sklearn.linear_model:Ridge", "regression", {}), Ridge
    )

    @register_model("tiny_forest", "classification")
    def tiny_forest(**params: object) -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=3, **params)  # type: ignore[arg-type]

    assert "tiny_forest" in available_models("classification")["classification"]
    assert create_model("tiny_forest", "classification", {}).n_estimators == 3


def test_available_models() -> None:
    models = available_models()
    assert "logistic_regression" in models["classification"]
    assert "ridge" in models["regression"]


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
