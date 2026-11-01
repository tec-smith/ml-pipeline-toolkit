"""The ``mlpipeline`` command.

    mlpipeline run CONFIG             train, evaluate and save a bundle
    mlpipeline check CONFIG           validate a config (and its data) without training
    mlpipeline evaluate BUNDLE DATA   score a saved model on new labelled data
    mlpipeline predict BUNDLE DATA    batch predictions to CSV, Parquet or JSON lines
    mlpipeline serve BUNDLE           run the prediction API
    mlpipeline info BUNDLE            show a bundle's metadata
    mlpipeline models                 list the built-in model names

Exit codes: 0 on success, 1 when the run fails, 2 for a bad config or bad
arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mlpipeline import __version__, _logging
from mlpipeline.artifacts import BundleError, load_bundle
from mlpipeline.config import ConfigError, PipelineConfig
from mlpipeline.data import DataError, load_table, prepare
from mlpipeline.evaluation import is_better
from mlpipeline.models import available_models, create_model

log = logging.getLogger("mlpipeline")

EXIT_FAILED = 1
EXIT_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _logging.configure(args.log_level, args.log_format)
    try:
        code: int = args.handler(args)
        return code
    except ConfigError as exc:
        log.error("invalid config: %s", exc)
        return EXIT_USAGE
    except (DataError, BundleError) as exc:
        log.error("%s", exc)
        return EXIT_FAILED
    except KeyboardInterrupt:
        return 130


def _run(args: argparse.Namespace) -> int:
    from mlpipeline.pipeline import run_pipeline

    config = PipelineConfig.from_yaml(args.config, args.set)
    result = run_pipeline(config, args.output_dir)
    if args.json:
        print(json.dumps(result.bundle.metadata["evaluation"], indent=2))
        return 0
    print(
        f"\n{config.name}: {config.model.type} ({config.task}), {result.seconds:.1f}s"
    )
    baseline = result.baseline or {}
    print(f"\n  {'metric':<18} {'model':>10} {'baseline':>10}")
    for name, value in result.metrics.items():
        ref = baseline.get(name)
        mark = (
            ""
            if ref is None
            else ("  better" if is_better(name, value, ref) else "  WORSE")
        )
        ref_text = "" if ref is None else f"{ref:.4f}"
        print(f"  {name:<18} {value:>10.4f} {ref_text:>10}{mark}")
    print(f"\nBundle saved to {result.output_dir}/ (see report.md)")
    return 0


def _check(args: argparse.Namespace) -> int:
    config = PipelineConfig.from_yaml(args.config, args.set)
    create_model(config.model.type, config.task, config.model.params, config.seed)
    if not args.no_data:
        frame = load_table(config.data.source)
        _, _, schema = prepare(frame, config.data, config.task)
        kinds = ", ".join(f"{f.name} ({f.kind})" for f in schema.features)
        print(f"data: {len(frame)} rows; features: {kinds}")
        if schema.classes is not None:
            print(f"classes: {', '.join(map(str, schema.classes))}")
    print(f"{args.config}: OK")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from mlpipeline.pipeline import evaluate_bundle

    bundle = load_bundle(args.bundle)
    scores = evaluate_bundle(bundle, load_table(args.data))
    if args.json:
        print(json.dumps(scores, indent=2))
        return 0
    trained = bundle.metadata["evaluation"]["metrics"]
    print(f"\n  {'metric':<18} {'new data':>10} {'at training':>12}")
    for name, value in scores["metrics"].items():
        ref = trained.get(name)
        print(f"  {name:<18} {value:>10.4f} {'' if ref is None else f'{ref:.4f}':>12}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.bundle)
    frame = load_table(args.data)
    out = frame.copy()
    column = args.column or f"{bundle.schema.target}_predicted"
    out[column] = bundle.predict(frame)
    if args.proba:
        proba = bundle.predict_proba(frame)
        for label in proba.columns:
            out[f"{column}_proba_{label}"] = proba[label]
    _write_table(out, args.output)
    log.info("predicted %d row(s)", len(out))
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from mlpipeline.serving import create_app
    except ImportError:
        log.error(
            "serving needs the serve extra: pip install 'ml-pipeline-toolkit[serve]'"
        )
        return EXIT_FAILED
    if not args.bundle:
        raise ConfigError("give a bundle directory, or set MODEL_DIR")
    app = create_app(load_bundle(args.bundle), max_batch_size=args.max_batch_size)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=None,  # keep our handlers
        access_log=args.access_log,
        proxy_headers=False,
    )
    return 0


def _info(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.bundle)
    meta = dict(bundle.metadata)
    if args.json:
        print(json.dumps(meta, indent=2))
        return 0
    report = Path(args.bundle) / "report.md"
    if report.is_file():
        print(report.read_text(encoding="utf-8"))
    else:
        print(
            json.dumps(
                {k: meta[k] for k in ("name", "task", "created_at", "schema")}, indent=2
            )
        )
    return 0


def _models(args: argparse.Namespace) -> int:
    for task, names in available_models(args.task).items():
        print(f"{task}:")
        for name in names:
            print(f"  {name}")
    return 0


def _write_table(frame: Any, output: str | None) -> None:
    if output is None or output == "-":
        frame.to_csv(sys.stdout, index=False)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix == ".jsonl":
        frame.to_json(path, orient="records", lines=True)
    elif suffix in (".csv", ""):
        frame.to_csv(path, index=False)
    else:
        raise ConfigError(
            f"--output: unsupported file type {suffix}; use .csv, .parquet or .jsonl"
        )


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise SystemExit(
            f"mlpipeline: {name} must be an integer, got {value!r}"
        ) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlpipeline",
        description="End-to-end machine learning pipelines from one YAML file.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--log-level",
        default=_env("LOG_LEVEL", "info"),
        choices=_logging.LEVELS,
        help="default: $LOG_LEVEL or info",
    )
    parser.add_argument(
        "--log-format",
        default=_env("LOG_FORMAT", "text"),
        choices=_logging.FORMATS,
        help="default: $LOG_FORMAT or text",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    overrides = argparse.ArgumentParser(add_help=False)
    overrides.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set model.params.n_estimators=50 (repeatable)",
    )

    p = sub.add_parser(
        "run", parents=[overrides], help="train, evaluate and save a model bundle"
    )
    p.add_argument("config", help="pipeline config (YAML)")
    p.add_argument(
        "-o",
        "--output-dir",
        help="bundle directory (default: output.dir in the config)",
    )
    p.add_argument("--json", action="store_true", help="print the evaluation as JSON")
    p.set_defaults(handler=_run)

    p = sub.add_parser(
        "check", parents=[overrides], help="validate a config without training"
    )
    p.add_argument("config")
    p.add_argument("--no-data", action="store_true", help="don't load the data source")
    p.set_defaults(handler=_check)

    p = sub.add_parser("evaluate", help="score a saved model on new labelled data")
    p.add_argument("bundle")
    p.add_argument("data", help="CSV, TSV, Parquet or JSON with the target column")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_evaluate)

    p = sub.add_parser("predict", help="batch predictions for a file")
    p.add_argument("bundle")
    p.add_argument(
        "data", help="CSV, TSV, Parquet or JSON; extra columns are passed through"
    )
    p.add_argument(
        "-o", "--output", help=".csv, .parquet or .jsonl (default: CSV to stdout)"
    )
    p.add_argument(
        "--column", help="name of the prediction column (default: <target>_predicted)"
    )
    p.add_argument(
        "--proba", action="store_true", help="add a probability column per class"
    )
    p.set_defaults(handler=_predict)

    p = sub.add_parser("serve", help="run the prediction API")
    p.add_argument(
        "bundle",
        nargs="?",
        default=os.environ.get("MODEL_DIR"),
        help="default: $MODEL_DIR",
    )
    p.add_argument(
        "--host", default=_env("HOST", "127.0.0.1"), help="default: $HOST or 127.0.0.1"
    )
    p.add_argument(
        "--port",
        type=int,
        default=_env_int("PORT", 8000),
        help="default: $PORT or 8000",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=_env_int("MAX_BATCH_SIZE", 1000),
        help="records per request (default: $MAX_BATCH_SIZE or 1000)",
    )
    p.add_argument(
        "--access-log",
        action=argparse.BooleanOptionalAction,
        default=_env("ACCESS_LOG", "true").lower() in ("1", "true", "yes"),
        help="log each request (default: $ACCESS_LOG or on)",
    )
    p.set_defaults(handler=_serve)

    p = sub.add_parser("info", help="show a bundle's report and metadata")
    p.add_argument("bundle")
    p.add_argument(
        "--json", action="store_true", help="print the full metadata as JSON"
    )
    p.set_defaults(handler=_info)

    p = sub.add_parser("models", help="list the built-in model names")
    p.add_argument("--task", choices=("classification", "regression"))
    p.set_defaults(handler=_models)
    return parser


if __name__ == "__main__":
    sys.exit(main())

# ***
#   @COPYRIGHT (c) 2026. Thomas EC. Smith (https://www.TECSmith.info). Licensed under the MIT License.
# ****
