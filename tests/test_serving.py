from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlpipeline import ModelBundle, PipelineConfig, run_pipeline
from mlpipeline.serving import create_app

RECORD = {"tenure": 3, "charges": 70.5, "plan": "basic", "region": "north"}


@pytest.fixture
def bundle(config: PipelineConfig) -> ModelBundle:
    return run_pipeline(config).bundle


@pytest.fixture
def client(bundle: ModelBundle) -> TestClient:
    return TestClient(create_app(bundle, max_batch_size=3, max_body_bytes=2000))


def test_health_and_metadata(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "model": "churn"}
    meta = client.get("/metadata").json()
    assert meta["task"] == "classification"
    assert [f["name"] for f in meta["schema"]["features"]] == [
        "tenure",
        "charges",
        "plan",
        "region",
    ]
    assert "config" not in meta
    assert "accuracy" in meta["metrics"]


def test_predict(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={
            "records": [
                RECORD,
                {**RECORD, "tenure": 55, "plan": "pro", "charges": None},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "churn"
    assert all(p in ("yes", "no") for p in body["predictions"])
    assert len(body["probabilities"]) == 2
    for row in body["probabilities"]:
        assert set(row) == {"no", "yes"}
        assert sum(row.values()) == pytest.approx(1)


def test_unseen_category_is_accepted(client: TestClient) -> None:
    response = client.post(
        "/predict", json={"records": [{**RECORD, "plan": "enterprise"}]}
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("records", "expected"),
    [
        (
            [{**RECORD, "tenure": "three"}],
            {"record": 0, "field": "tenure", "error": "expected a number or null"},
        ),
        (
            [{**RECORD, "colour": "red"}],
            {"record": 0, "field": "colour", "error": "unknown field"},
        ),
        (
            [{k: v for k, v in RECORD.items() if k != "plan"}],
            {"record": 0, "field": "plan", "error": "missing field"},
        ),
    ],
)
def test_invalid_records(
    client: TestClient, records: list[dict[str, Any]], expected: dict[str, Any]
) -> None:
    response = client.post("/predict", json={"records": records})
    assert response.status_code == 422
    assert expected in response.json()["detail"]


def test_request_limits(client: TestClient) -> None:
    assert client.post("/predict", json={"records": []}).status_code == 422
    assert client.post("/predict", json={"records": [RECORD] * 4}).status_code == 413
    big = client.post("/predict", json={"records": [{**RECORD, "plan": "x" * 3000}]})
    assert big.status_code == 413


def test_regression_has_no_probabilities() -> None:
    config = PipelineConfig.from_dict(
        {
            "name": "diabetes",
            "task": "regression",
            "data": {
                "source": "sklearn:diabetes",
                "target": "target",
                "features": ["bmi", "bp"],
            },
            "model": {"type": "ridge"},
            "evaluation": {"cv_folds": 0},
        }
    )
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        bundle = run_pipeline(config, tmp + "/b").bundle
    client = TestClient(create_app(bundle))
    body = client.post("/predict", json={"records": [{"bmi": 0.05, "bp": 0.02}]}).json()
    assert isinstance(body["predictions"][0], float)
    assert "probabilities" not in body


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
