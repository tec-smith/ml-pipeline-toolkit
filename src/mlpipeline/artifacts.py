"""Model bundles: a fitted pipeline plus everything needed to trust and serve it.

A bundle is a directory::

    model.joblib     the fitted scikit-learn pipeline, preprocessing included
    metadata.json    schema, metrics, config, versions and the model's SHA-256
    report.md        a readable summary of the run

Loading checks the checksum before unpickling anything, and warns when the
installed scikit-learn differs from the one that trained the model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from mlpipeline.data import Schema

log = logging.getLogger(__name__)

BUNDLE_FORMAT = 1
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REPORT_FILE = "report.md"


class BundleError(Exception):
    """A model bundle is missing, incomplete, corrupted or incompatible."""


@dataclass
class ModelBundle:
    pipeline: Any
    metadata: dict[str, Any]
    path: Path | None = None

    @property
    def name(self) -> str:
        return str(self.metadata["name"])

    @property
    def schema(self) -> Schema:
        return Schema.from_dict(self.metadata["schema"])

    @property
    def can_predict_proba(self) -> bool:
        return self.schema.task == "classification" and hasattr(
            self.pipeline, "predict_proba"
        )

    def predict(self, frame: pd.DataFrame, strict: bool = False) -> np.ndarray:
        """Predict for each row. Extra columns are ignored unless ``strict``."""
        return np.asarray(self.pipeline.predict(self.schema.check_frame(frame, strict)))

    def predict_proba(self, frame: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """Class probabilities, one column per class."""
        if not self.can_predict_proba:
            raise BundleError(f"model {self.name!r} doesn't predict probabilities")
        proba = self.pipeline.predict_proba(self.schema.check_frame(frame, strict))
        classes = [str(c) for c in self.pipeline.classes_]
        return pd.DataFrame(proba, columns=classes, index=frame.index)


def save_bundle(
    bundle: ModelBundle, directory: str | Path, report: str | None = None
) -> Path:
    """Write a bundle, replacing any bundle already in ``directory``.

    Files are written to a temporary directory next to the target and moved
    into place at the end, so a reader never sees a half-written bundle.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        model_path = staging / MODEL_FILE
        joblib.dump(bundle.pipeline, model_path, compress=3)
        metadata = {**bundle.metadata, "bundle_format": BUNDLE_FORMAT}
        metadata["model_sha256"] = _sha256(model_path)
        (staging / METADATA_FILE).write_text(
            json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8"
        )
        if report is not None:
            (staging / REPORT_FILE).write_text(report, encoding="utf-8")
        _replace_dir(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    bundle.metadata = metadata
    bundle.path = directory
    log.info("saved bundle to %s", directory)
    return directory


def load_bundle(directory: str | Path) -> ModelBundle:
    """Read a bundle written by :func:`save_bundle`.

    Only load bundles you trust: ``model.joblib`` is a pickle, and unpickling
    runs code. The checksum catches corruption and accidental edits, not a
    deliberate attacker who can also rewrite ``metadata.json``.
    """
    directory = Path(directory)
    meta_path = directory / METADATA_FILE
    model_path = directory / MODEL_FILE
    if not meta_path.is_file() or not model_path.is_file():
        raise BundleError(
            f"{directory}: not a model bundle (needs {METADATA_FILE} and {MODEL_FILE})"
        )
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleError(f"{meta_path}: invalid JSON: {exc}") from exc
    if metadata.get("bundle_format") != BUNDLE_FORMAT:
        raise BundleError(
            f"{directory}: bundle format {metadata.get('bundle_format')!r} is not "
            f"supported (this version reads format {BUNDLE_FORMAT})"
        )
    expected = metadata.get("model_sha256")
    actual = _sha256(model_path)
    if expected != actual:
        raise BundleError(
            f"{model_path}: checksum mismatch, the file is corrupted or was changed"
        )

    trained_with = metadata.get("versions", {}).get("scikit-learn")
    if trained_with and trained_with != sklearn.__version__:
        log.warning(
            "model was trained with scikit-learn %s but %s is installed; "
            "predictions may differ",
            trained_with,
            sklearn.__version__,
        )
    with warnings.catch_warnings():
        # Already reported above, once and more readably.
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        pipeline = joblib.load(model_path)
    return ModelBundle(pipeline=pipeline, metadata=metadata, path=directory)


def _replace_dir(new: Path, target: Path) -> None:
    if not target.exists():
        os.replace(new, target)
        return
    if not (target / METADATA_FILE).is_file() and any(target.iterdir()):
        raise BundleError(f"{target} exists and isn't a model bundle; not replacing it")
    old = target.with_name(f".{target.name}.old")
    shutil.rmtree(old, ignore_errors=True)
    os.replace(target, old)
    os.replace(new, target)
    shutil.rmtree(old, ignore_errors=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
