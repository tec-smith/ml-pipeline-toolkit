"""Use the toolkit as a library instead of from the command line.

uv run python examples/python_api.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlpipeline import PipelineConfig, load_bundle, register_model, run_pipeline


# Any scikit-learn compatible estimator can be registered under a name and
# then used as model.type in a config.
@register_model("balanced_forest", "classification")
def balanced_forest(**params: object) -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", **params)  # type: ignore[arg-type]


config = PipelineConfig.from_yaml(
    "examples/churn.yml",
    overrides=[
        "name=churn-balanced",
        "model.type=balanced_forest",
        "model.params={n_estimators: 300, min_samples_leaf: 5}",
        "model.search=null",
    ],
)
result = run_pipeline(config, output_dir="artifacts/churn-balanced")
baseline = result.baseline or {}
print(f"recall {result.metrics['recall']:.3f} (baseline {baseline['recall']:.3f})")

# Load it back the way a separate prediction job would.
bundle = load_bundle(result.output_dir)
customers = pd.DataFrame(
    [
        {
            "tenure_months": 3,
            "monthly_charges": 89.9,
            "total_charges": 270.0,
            "contract": "month-to-month",
            "internet_service": "fiber",
            "payment_method": "electronic-check",
            "support_tickets": 3,
            "senior_citizen": "no",
        },
        {
            "tenure_months": 60,
            "monthly_charges": 30.0,
            "total_charges": None,  # missing values are imputed
            "contract": "two-year",
            "internet_service": "none",
            "payment_method": "bank-transfer",
            "support_tickets": 0,
            "senior_citizen": "no",
        },
    ]
)
print(bundle.predict(customers))
print(bundle.predict_proba(customers).round(3))

# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
