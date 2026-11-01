"""The model registry: names in a config mapped to scikit-learn estimators.

Built-in names cover the common estimators. Register your own with
:func:`register_model`, or name any estimator class directly in a config as
``package.module:ClassName``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any, overload

from sklearn import ensemble, linear_model, naive_bayes, neighbors, svm, tree

from mlpipeline.config import ConfigError, Task

Factory = Callable[..., Any]

_REGISTRY: dict[Task, dict[str, Factory]] = {"classification": {}, "regression": {}}


@overload
def register_model(name: str, task: Task, factory: Factory) -> Factory: ...
@overload
def register_model(name: str, task: Task) -> Callable[[Factory], Factory]: ...
def register_model(
    name: str, task: Task, factory: Factory | None = None
) -> Factory | Callable[[Factory], Factory]:
    """Make ``name`` usable as ``model.type`` for ``task``.

    ``factory`` is called with the config's ``model.params`` as keyword
    arguments and must return a scikit-learn compatible estimator. Use it
    directly or as a decorator::

        @register_model("tuned_forest", "classification")
        def tuned_forest(**params):
            return RandomForestClassifier(n_estimators=500, **params)
    """

    def add(f: Factory) -> Factory:
        _REGISTRY[task][name] = f
        return f

    return add if factory is None else add(factory)


def available_models(task: Task | None = None) -> dict[Task, list[str]]:
    """The registered model names, per task."""
    tasks: tuple[Task, ...] = (task,) if task else ("classification", "regression")
    return {t: sorted(_REGISTRY[t]) for t in tasks}


def create_model(
    name: str, task: Task, params: dict[str, Any], seed: int | None = None
) -> Any:
    """Build an unfitted estimator, seeding it if it takes a ``random_state``."""
    factory = _resolve(name, task)
    try:
        model = factory(**params)
    except TypeError as exc:
        raise ConfigError(f"model.params: {exc}") from exc
    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise ConfigError(
            f"model.type {name!r} did not return an estimator with fit and predict"
        )
    if (
        seed is not None
        and "random_state" not in params
        and hasattr(model, "get_params")
        and "random_state" in model.get_params(deep=False)
    ):
        model.set_params(random_state=seed)
    return model


def _resolve(name: str, task: Task) -> Factory:
    if name in _REGISTRY[task]:
        return _REGISTRY[task][name]
    if ":" in name:
        module_name, _, attr = name.partition(":")
        try:
            obj = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"model.type {name!r}: can't import ({exc})") from exc
        if not callable(obj):
            raise ConfigError(f"model.type {name!r} is not callable")
        return obj  # type: ignore[no-any-return]
    other: Task = "regression" if task == "classification" else "classification"
    hint = f" ({name!r} is a {other} model)" if name in _REGISTRY[other] else ""
    raise ConfigError(
        f"model.type: unknown {task} model {name!r}{hint}; "
        f"use one of {', '.join(sorted(_REGISTRY[task]))}, or package.module:Class"
    )


def _with_defaults(cls: type[Any], **defaults: Any) -> Factory:
    def factory(**params: Any) -> Any:
        return cls(**{**defaults, **params})

    factory.__doc__ = inspect.getdoc(cls)
    return factory


for _name, _factory in {
    "logistic_regression": _with_defaults(
        linear_model.LogisticRegression, max_iter=1000
    ),
    "random_forest": _with_defaults(ensemble.RandomForestClassifier),
    "extra_trees": _with_defaults(ensemble.ExtraTreesClassifier),
    "gradient_boosting": _with_defaults(ensemble.GradientBoostingClassifier),
    "hist_gradient_boosting": _with_defaults(ensemble.HistGradientBoostingClassifier),
    "decision_tree": _with_defaults(tree.DecisionTreeClassifier),
    "knn": _with_defaults(neighbors.KNeighborsClassifier),
    "svm": _with_defaults(svm.SVC, probability=True),
    "naive_bayes": _with_defaults(naive_bayes.GaussianNB),
}.items():
    register_model(_name, "classification", _factory)

for _name, _factory in {
    "linear_regression": _with_defaults(linear_model.LinearRegression),
    "ridge": _with_defaults(linear_model.Ridge),
    "lasso": _with_defaults(linear_model.Lasso),
    "elastic_net": _with_defaults(linear_model.ElasticNet),
    "random_forest": _with_defaults(ensemble.RandomForestRegressor),
    "extra_trees": _with_defaults(ensemble.ExtraTreesRegressor),
    "gradient_boosting": _with_defaults(ensemble.GradientBoostingRegressor),
    "hist_gradient_boosting": _with_defaults(ensemble.HistGradientBoostingRegressor),
    "decision_tree": _with_defaults(tree.DecisionTreeRegressor),
    "knn": _with_defaults(neighbors.KNeighborsRegressor),
    "svr": _with_defaults(svm.SVR),
}.items():
    register_model(_name, "regression", _factory)

# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
