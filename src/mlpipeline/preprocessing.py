"""Turning raw columns into model inputs.

Everything here becomes a step of the saved scikit-learn pipeline, so the
same transformations run in training, batch prediction and the API without
any code outside the bundle.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from mlpipeline.config import PreprocessingConfig
from mlpipeline.data import DataError, Schema

MISSING = "__missing__"

_SCALERS: dict[str, type[Any]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class SchemaCoercer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Select the schema's columns and give each a consistent type.

    Numeric columns become floats, with a clear error for values that aren't
    numbers. Categorical values become strings, so ``1``, ``1.0`` and ``"1"``
    from a CSV, a JSON request or a DataFrame all mean the same category.
    Missing values stay missing for the imputers to fill.
    """

    def __init__(
        self, numeric: list[str] | None = None, categorical: list[str] | None = None
    ):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X: pd.DataFrame, y: Any = None) -> SchemaCoercer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = self.numeric or []
        categorical = self.categorical or []
        out = pd.DataFrame(index=X.index)
        for name in numeric:
            try:
                out[name] = pd.to_numeric(X[name], errors="raise").astype("float64")
            except (TypeError, ValueError) as exc:
                raise DataError(f"column {name!r}: expected numbers ({exc})") from exc
        for name in categorical:
            out[name] = X[name].map(_category, na_action="ignore").astype(object)
            out[name] = out[name].where(out[name].notna(), np.nan)
        return out

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(
            [*(self.numeric or []), *(self.categorical or [])], dtype=object
        )


def build_preprocessor(schema: Schema, cfg: PreprocessingConfig) -> Pipeline:
    numeric = schema.of_kind("numeric")
    categorical = schema.of_kind("categorical")
    branches: list[tuple[str, Any, list[str]]] = []
    if numeric:
        branches.append(("numeric", _numeric_branch(cfg), numeric))
    if categorical:
        branches.append(("categorical", _categorical_branch(cfg), categorical))
    columns = ColumnTransformer(branches, remainder="drop", sparse_threshold=0.0)
    return Pipeline(
        [
            ("coerce", SchemaCoercer(numeric, categorical)),
            ("columns", columns),
        ]
    )


def _numeric_branch(cfg: PreprocessingConfig) -> Pipeline:
    impute = cfg.numeric.impute
    imputer = (
        SimpleImputer(strategy="constant", fill_value=0.0)
        if impute == "zero"
        else SimpleImputer(strategy=impute)
    )
    steps: list[tuple[str, Any]] = [("impute", imputer)]
    if cfg.numeric.scale != "none":
        steps.append(("scale", _SCALERS[cfg.numeric.scale]()))
    return Pipeline(steps)


def _categorical_branch(cfg: PreprocessingConfig) -> Pipeline:
    c = cfg.categorical
    imputer = (
        SimpleImputer(strategy="constant", fill_value=MISSING)
        if c.impute == "constant"
        else SimpleImputer(strategy="most_frequent")
    )
    encoder: Any
    if c.encode == "onehot":
        grouped = c.max_categories is not None or c.min_frequency is not None
        encoder = OneHotEncoder(
            # Unseen categories at prediction time join the "infrequent"
            # bucket if there is one, or encode as all zeros.
            handle_unknown="infrequent_if_exist" if grouped else "ignore",
            max_categories=c.max_categories,
            min_frequency=c.min_frequency,
            sparse_output=False,
        )
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return Pipeline([("impute", imputer), ("encode", encoder)])


def _category(value: Any) -> str:
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
