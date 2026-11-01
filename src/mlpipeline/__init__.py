"""End-to-end machine learning pipelines from one YAML file.

A run loads a table, infers its schema, preprocesses it, trains and tunes a
scikit-learn model, evaluates it against a baseline, and saves everything
needed to serve it as a versioned bundle::

    from mlpipeline import PipelineConfig, load_bundle, run_pipeline

    result = run_pipeline(PipelineConfig.from_yaml("examples/churn.yml"))
    bundle = load_bundle(result.output_dir)
    bundle.predict(new_customers)
"""

from mlpipeline.artifacts import BundleError, ModelBundle, load_bundle, save_bundle
from mlpipeline.config import ConfigError, PipelineConfig
from mlpipeline.data import DataError, Feature, Schema
from mlpipeline.models import available_models, create_model, register_model
from mlpipeline.pipeline import RunResult, evaluate_bundle, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "BundleError",
    "ConfigError",
    "DataError",
    "Feature",
    "ModelBundle",
    "PipelineConfig",
    "RunResult",
    "Schema",
    "__version__",
    "available_models",
    "create_model",
    "evaluate_bundle",
    "load_bundle",
    "register_model",
    "run_pipeline",
    "save_bundle",
]

# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
