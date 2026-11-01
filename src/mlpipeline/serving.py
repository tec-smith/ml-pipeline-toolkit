"""An HTTP prediction service for a model bundle.

Needs the ``serve`` extra: ``pip install 'ml-pipeline-toolkit[serve]'``.

    GET  /health     liveness and readiness: the model is loaded
    GET  /metadata   the model's name, schema and test metrics
    POST /predict    {"records": [{...}, ...]} -> predictions
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mlpipeline import __version__
from mlpipeline.artifacts import ModelBundle
from mlpipeline.data import DataError

log = logging.getLogger(__name__)

DEFAULT_MAX_BATCH = 1000
DEFAULT_MAX_BODY_BYTES = 5 * 1024 * 1024


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(
        min_length=1, description="One object per row"
    )


class PredictResponse(BaseModel):
    model: str
    predictions: list[Any]
    probabilities: list[dict[str, float]] | None = None


def create_app(
    bundle: ModelBundle,
    max_batch_size: int = DEFAULT_MAX_BATCH,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> FastAPI:
    """Build the app for one loaded bundle."""
    schema = bundle.schema
    app = FastAPI(
        title=f"{bundle.name} prediction API",
        version=__version__,
        description=f"Serves the `{bundle.name}` {schema.task} model.",
    )

    @app.middleware("http")
    async def limit_body(request: Request, call_next: Any) -> Any:
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > max_body_bytes:
            return JSONResponse(
                {"detail": f"request body is larger than {max_body_bytes} bytes"},
                status_code=413,
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": bundle.name}

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        meta = bundle.metadata
        # The training config and data source path are left out: they describe
        # your infrastructure, which API clients don't need to see.
        return {
            "name": meta["name"],
            "task": meta["task"],
            "created_at": meta["created_at"],
            "toolkit_version": meta.get("toolkit_version"),
            "model_type": meta["model"]["type"],
            "schema": meta["schema"],
            "metrics": meta["evaluation"]["metrics"],
        }

    # A plain def, so FastAPI runs the model in its thread pool rather than
    # blocking the event loop.
    @app.post(
        "/predict", response_model=PredictResponse, response_model_exclude_none=True
    )
    def predict(body: PredictRequest) -> PredictResponse:
        if len(body.records) > max_batch_size:
            raise HTTPException(413, f"at most {max_batch_size} records per request")
        errors = schema.check_records(body.records)
        if errors:
            raise HTTPException(422, errors[:100])
        frame = pd.DataFrame.from_records(body.records, columns=schema.names)
        try:
            predictions = bundle.predict(frame)
            proba = bundle.predict_proba(frame) if bundle.can_predict_proba else None
        except DataError as exc:
            raise HTTPException(422, str(exc)) from exc
        return PredictResponse(
            model=bundle.name,
            predictions=[p.item() if hasattr(p, "item") else p for p in predictions],
            probabilities=None
            if proba is None
            else [
                {str(k): float(v) for k, v in row.items()}
                for row in proba.to_dict(orient="records")
            ],
        )

    log.info(
        "serving %r (%s, %d features)", bundle.name, schema.task, len(schema.features)
    )
    return app


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
