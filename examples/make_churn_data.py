"""Generate examples/data/churn.csv, a synthetic customer churn table.

It has the things real tables have: numeric and text columns, an ID column
to drop, missing values and a class imbalance. The same seed always produces
the same file.

    python examples/make_churn_data.py [ROWS]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7


def make(rows: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], rows, p=[0.55, 0.25, 0.2]
    )
    internet = rng.choice(["fiber", "dsl", "none"], rows, p=[0.45, 0.35, 0.2])
    payment = rng.choice(
        ["electronic-check", "card", "bank-transfer", "mailed-check"],
        rows,
        p=[0.35, 0.25, 0.25, 0.15],
    )
    tenure = rng.integers(1, 73, rows)
    base_charge = np.select(
        [internet == "fiber", internet == "dsl"], [80.0, 55.0], 25.0
    )
    monthly = np.round(base_charge + rng.normal(0, 12, rows).clip(-20, 30), 2)
    total = np.round(monthly * tenure * rng.uniform(0.95, 1.05, rows), 2)
    tickets = rng.poisson(1.2, rows)
    senior = rng.choice(["yes", "no"], rows, p=[0.16, 0.84])

    logit = (
        -1.4
        + 1.3 * (contract == "month-to-month")
        - 0.9 * (contract == "two-year")
        + 0.6 * (internet == "fiber")
        + 0.5 * (payment == "electronic-check")
        - 0.035 * tenure
        + 0.012 * (monthly - 60)
        + 0.35 * tickets
        + 0.3 * (senior == "yes")
    )
    churned = rng.random(rows) < 1 / (1 + np.exp(-logit))

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(1, rows + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "support_tickets": tickets,
            "senior_citizen": senior,
            "churned": np.where(churned, "yes", "no"),
        }
    )
    # Real exports have gaps.
    frame.loc[rng.random(rows) < 0.03, "total_charges"] = np.nan
    frame.loc[rng.random(rows) < 0.02, "payment_method"] = np.nan
    return frame


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = Path(__file__).parent / "data" / "churn.csv"
    out.parent.mkdir(exist_ok=True)
    make(rows).to_csv(out, index=False)
    print(f"wrote {rows} rows to {out}")

# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
