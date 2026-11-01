# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may
change the API, the config format or the bundle format.

## [Unreleased]

## [0.1.0] - 2026-11-01

First release.

### Added

- YAML pipeline configs, validated up front, with `--set` overrides.
- Data loading from CSV, TSV, Parquet, JSON, JSON lines and the datasets
  bundled with scikit-learn, with schema inference and seeded, stratified
  train/test splits.
- Preprocessing: numeric imputation and scaling, and categorical imputation
  with one-hot or ordinal encoding that handles rare and unseen categories.
- A model registry with 20 built-in classification and regression models,
  `register_model`, and `package.module:Class` import paths.
- Grid and random hyperparameter search, and cross-validation.
- Evaluation on a held-out test set against a baseline, with a confusion
  matrix and a Markdown report.
- Model bundles (format 1) with metadata, library versions and a SHA-256
  checksum, written atomically.
- A FastAPI prediction service with schema validation and size limits.
- The `mlpipeline` command: `run`, `check`, `evaluate`, `predict`, `serve`,
  `info` and `models`.
- Churn, iris and diabetes examples, a Docker image and a Docker Compose demo.

[Unreleased]: https://github.com/tec-smith/ml-pipeline-toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tec-smith/ml-pipeline-toolkit/releases/tag/v0.1.0
