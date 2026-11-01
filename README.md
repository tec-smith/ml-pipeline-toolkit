# ML Pipeline Toolkit

An end-to-end machine learning pipeline toolkit for Python, with data
preprocessing, model training, evaluation and deployment utilities. Describe
a run in one YAML file, and the toolkit loads the data, trains and tunes a
scikit-learn model, checks it against a baseline, and saves a versioned
bundle you can serve as an HTTP API or use for batch predictions.

[![CI](https://github.com/tec-smith/ml-pipeline-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/tec-smith/ml-pipeline-toolkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

More about this project: [tecsmith.info/portfolio/repositories/ml-pipeline-toolkit](https://www.tecsmith.info/portfolio/repositories/ml-pipeline-toolkit)

## When to use this

The toolkit covers the path from a table of data to a model behind an API.
It uses scikit-learn for modelling and doesn't wrap it in its own
abstractions: what you save and serve is an ordinary scikit-learn `Pipeline`.

**It's a good fit for:**

- Tabular problems (classification and regression) where scikit-learn models
  are the right tool, which covers most business prediction tasks.
- Teams that want training to be repeatable and reviewable: the YAML config
  goes in version control, and the same config and seed give the same model.
- Getting a model into production without writing glue code: the bundle
  holds the preprocessing, so the API and batch jobs can't apply different
  transformations from training.
- Learning how a production ML pipeline fits together: each stage is one
  short, readable module.

**Use something else when:**

- You're training deep learning models on images, text or audio. Use
  PyTorch or TensorFlow, with a serving layer such as TorchServe or Triton.
- You need experiment tracking across hundreds of runs, a model registry with
  approvals, or a feature store. That's MLflow, Weights & Biases or a
  platform such as SageMaker or Vertex AI (this toolkit's bundles work well
  as the thing you log into one of them).
- Your data doesn't fit in memory on one machine. Look at Spark MLlib or
  Dask-ML.

## What's included

| Area             | What you get                                                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Configuration    | One YAML file per pipeline, validated up front with clear errors, plus `--set key=value` overrides from the command line              |
| Data loading     | CSV, TSV, Parquet, JSON and JSON lines, and datasets bundled with scikit-learn for trying things out                                  |
| Schema           | Numeric and categorical columns inferred (or set by you), saved with the model and enforced on every prediction request               |
| Preprocessing    | Imputation, scaling and one-hot or ordinal encoding, with rare and unseen categories handled, all inside the saved pipeline           |
| Training         | 20 built-in models, your own via `register_model` or an import path, seeded for reproducibility, grid or random hyperparameter search |
| Evaluation       | Held-out test metrics, cross-validation on the training set, a baseline to compare with, a confusion matrix and a Markdown report      |
| Model bundles    | The fitted pipeline, metadata, schema, metrics and library versions in one directory, written atomically and checked with SHA-256     |
| Batch prediction | `mlpipeline predict` for files, with class probabilities and your other columns passed through                                        |
| Serving          | A FastAPI service with input validation, batch and body size limits, OpenAPI docs, and health and metadata endpoints                  |
| Operations       | A Docker image that runs as a non-root user, a Compose demo, JSON logs, GitHub Actions CI and Dependabot                              |

## Quick start

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/). Docker is
optional, for the demo stack.

```bash
git clone https://github.com/tec-smith/ml-pipeline-toolkit.git
cd ml-pipeline-toolkit

make install   # creates .venv with everything needed
make train     # trains examples/churn.yml into artifacts/churn
make serve     # serves it on localhost:8000
```

Training prints the test metrics next to a baseline that ignores the
features, so you can see at a glance whether the model learned anything:

```text
churn: hist_gradient_boosting (classification), 6.8s

  metric                  model   baseline
  accuracy               0.7450     0.7100  better
  balanced_accuracy      0.6215     0.5000  better
  precision              0.6129     0.0000  better
  recall                 0.3276     0.0000  better
  f1                     0.4270     0.0000  better
  log_loss               0.5162     0.6022  better
  roc_auc                0.7518     0.5000  better

Bundle saved to artifacts/churn/ (see report.md)
```

Then, in another terminal:

```bash
curl localhost:8000/metadata   # schema and metrics

curl localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "records": [{
    "tenure_months": 2, "monthly_charges": 95.5, "total_charges": 190,
    "contract": "month-to-month", "internet_service": "fiber",
    "payment_method": "electronic-check", "support_tickets": 4,
    "senior_citizen": "yes"
  }]
}'
# {"model":"churn","predictions":["yes"],"probabilities":[{"no":0.13,"yes":0.87}]}
```

Interactive API docs are at [localhost:8000/docs](http://localhost:8000/docs).

> **Prefer Docker?** `make demo` trains the model in one container and serves
> it from another, on the same port.
>
> **Prefer pip?** `pip install -e ".[serve]"` in a virtual environment, then
> use `mlpipeline` directly: `mlpipeline run examples/churn.yml`.

## Commands

| Command          | What it does                                                    |
| ---------------- | --------------------------------------------------------------- |
| `make install`   | Create `.venv` with the toolkit, the serve extra and dev tools  |
| `make train`     | Train `CONFIG` (default `examples/churn.yml`)                   |
| `make serve`     | Serve `BUNDLE` (default `artifacts/churn`) on port 8000         |
| `make demo`      | Train and serve the churn model with Docker Compose             |
| `make validate`  | Format check, lint, type check and tests: run before you push   |
| `make test`      | Run the tests                                                   |
| `make coverage`  | Tests with a coverage report in `htmlcov/index.html`            |
| `make fmt`       | Format every Python file (`fmt-check` to verify)                |
| `make docker`    | Build the Docker image                                          |
| `make data`      | Regenerate the synthetic `examples/data/churn.csv`              |

The `mlpipeline` command does the work behind them:

| Command                             | What it does                                                 |
| ----------------------------------- | ------------------------------------------------------------ |
| `mlpipeline run CONFIG`             | Train, evaluate and save a bundle                            |
| `mlpipeline check CONFIG`           | Validate a config, its model and its data without training   |
| `mlpipeline evaluate BUNDLE DATA`   | Score a saved model on new labelled data                     |
| `mlpipeline predict BUNDLE DATA`    | Batch predictions to CSV, Parquet or JSON lines              |
| `mlpipeline serve [BUNDLE]`         | Run the prediction API                                       |
| `mlpipeline info BUNDLE`            | Show a bundle's report, or its full metadata with `--json`   |
| `mlpipeline models`                 | List the built-in model names                                |

It exits with `0` on success, `1` when a run fails, and `2` for an invalid
config or arguments, so it slots into scripts and schedulers.

## Project structure

```text
├── src/mlpipeline/
│   ├── config.py            # The YAML config: dataclasses, validation, overrides
│   ├── data.py              # Loading tables, schema inference, train/test split
│   ├── preprocessing.py     # Imputation, scaling and encoding as pipeline steps
│   ├── models.py            # The model registry and register_model
│   ├── training.py          # Fitting, hyperparameter search, cross-validation
│   ├── evaluation.py        # Metrics, the baseline and the Markdown report
│   ├── artifacts.py         # Saving and loading model bundles
│   ├── pipeline.py          # run_pipeline: one call from config to bundle
│   ├── serving.py           # The FastAPI prediction service
│   └── cli.py               # The mlpipeline command
├── examples/
│   ├── churn.yml            # Binary classification on a CSV, with a grid search
│   ├── iris.yml             # Multiclass classification, the smallest config
│   ├── diabetes.yml         # Regression with a random search
│   ├── python_api.py        # Using the toolkit as a library
│   ├── make_churn_data.py   # Generates the synthetic churn dataset
│   └── data/churn.csv
├── tests/                   # Unit, integration, API and CLI tests
├── Dockerfile               # Image that trains or serves
├── docker-compose.yml       # Demo stack: train, then serve
└── .github/                 # CI workflow, Dependabot, issue and PR templates
```

## How it works

```text
  config.yml ──► load ──► prepare ──► split ──► train ──────────► evaluate ──► bundle/
                 CSV,     schema,     train /   preprocessing     test set,      model.joblib
                 Parquet  features,   test      + model,          baseline,      metadata.json
                 ...      target      (seeded)  search, CV        report         report.md
                                                                                    │
                                     ┌──────────────────────────────────────────────┤
                                     ▼                                              ▼
                       mlpipeline serve (HTTP)                       mlpipeline predict (files)
                       validate against schema → pipeline.predict
```

- **Configuration.** The YAML file is parsed into frozen dataclasses before
  anything else happens. Unknown keys, wrong types and invalid choices are
  errors that name the exact key, such as
  `preprocessing.numeric.scale: expected one of standard, minmax, robust, none`.
  `mlpipeline check` also builds the model and loads the data, so a bad
  parameter or a missing column fails in seconds, not after training.
- **Schema.** Every column except the target and any you `drop` becomes a
  feature. Numbers are numeric, and everything else (text, booleans) is
  categorical. List columns under `data.numeric` or `data.categorical` to
  override that, for example for a numeric product code. Rows with no target
  value are dropped with a warning.
- **Splitting.** A seeded train/test split, stratified by class for
  classification so the test set has the same class balance as the data.
  The test set isn't touched until evaluation.
- **Preprocessing.** Numeric columns are imputed (median by default) and
  scaled. Categorical columns are imputed with a `__missing__` category and
  one-hot encoded, with categories beyond the 20 most frequent grouped as
  "infrequent". Values are normalised first, so `1`, `1.0` and `"1"` are the
  same category whether they come from a CSV, a DataFrame or JSON. Categories
  never seen in training are accepted at prediction time and encoded as
  infrequent (or all zeros).
- **Training.** The preprocessing and the model form one scikit-learn
  `Pipeline`, so cross-validation and the search fit the preprocessing on
  each training fold only, and nothing leaks from the validation fold. The
  `seed` is passed to the split, the folds, the search and any model that
  takes a `random_state`.
- **Search.** Parameters under `model.search.params` are tried with 5-fold
  cross-validation, and the best combination is refitted on the whole
  training set. Grid search tries every combination, and random search
  samples `n_iter` of them.
- **Evaluation.** The fitted pipeline is scored on the test set, and so is a
  baseline that ignores the features (the class priors, or the mean). For
  binary problems, precision, recall and F1 are for the positive class, which
  is `data.positive_class` or the last class in sorted order. For multiclass
  problems they're macro averages. Cross-validation scores on the training
  set show how stable the model is.
- **Bundles.** The pipeline is saved with joblib, next to `metadata.json`,
  which records the schema, metrics, config, library versions and the
  model's SHA-256 checksum. The bundle is written to a temporary directory and
  moved into place, so a process loading it at the same moment never sees
  half a bundle. Loading verifies the checksum before unpickling anything and warns
  if the installed scikit-learn differs from the one used in training.

### Built-in models

Run `mlpipeline models` for the list. Any scikit-learn parameter of the model
can go under `model.params`.

| Classification           | Regression                                                     |
| ------------------------ | -------------------------------------------------------------- |
| `logistic_regression`    | `linear_regression`, `ridge`, `lasso`, `elastic_net`           |
| `random_forest`          | `random_forest`                                                |
| `extra_trees`            | `extra_trees`                                                  |
| `gradient_boosting`      | `gradient_boosting`                                            |
| `hist_gradient_boosting` | `hist_gradient_boosting`                                       |
| `decision_tree`          | `decision_tree`                                                |
| `knn`                    | `knn`                                                          |
| `svm`                    | `svr`                                                          |
| `naive_bayes`            |                                                                |

### Metrics

| Task           | Test set and baseline                                                         | Cross-validation               |
| -------------- | ----------------------------------------------------------------------------- | ------------------------------ |
| Classification | accuracy, balanced accuracy, precision, recall, F1, log loss, ROC AUC         | accuracy, balanced accuracy, F1 |
| Regression     | MAE, RMSE, median absolute error, R², MAPE (when no true value is zero)       | MAE, RMSE, R²                  |

## Using it as a library

**Run a pipeline.** Everything the command line does is one function call:

```python
from mlpipeline import PipelineConfig, run_pipeline

config = PipelineConfig.from_yaml("examples/churn.yml", overrides=["seed=7"])
result = run_pipeline(config)
print(result.metrics["roc_auc"], result.baseline["roc_auc"])
```

Pass `frame=` to train on a DataFrame you already have instead of reading
`data.source`.

**Predict.** Load a bundle in any process that has the toolkit installed:

```python
from mlpipeline import load_bundle

bundle = load_bundle("artifacts/churn")
bundle.predict(customers)  # numpy array of labels
bundle.predict_proba(customers)  # DataFrame, one column per class
bundle.schema.names  # the columns it needs
```

`bundle.pipeline` is the plain scikit-learn `Pipeline`, if you'd rather use
it directly.

**Bring your own model.** Register a factory once, then use its name in any
config:

```python
from mlpipeline import register_model
from sklearn.ensemble import RandomForestClassifier


@register_model("balanced_forest", "classification")
def balanced_forest(**params):
    return RandomForestClassifier(class_weight="balanced", **params)
```

Or skip registration and name the class in the config, as
`type: sklearn.svm:LinearSVC` or `type: xgboost:XGBClassifier`. Anything with
scikit-learn's `fit` and `predict` works.

**Monitor a deployed model.** When outcomes arrive, score the bundle on them
and compare with the metrics from training:

```bash
mlpipeline evaluate artifacts/churn last_month.csv
```

See [examples/python_api.py](examples/python_api.py) for a complete script.

## Prediction API

`mlpipeline serve` serves one bundle on `127.0.0.1:8000` by default.

| Method | Path        | Does                                                                     |
| ------ | ----------- | ------------------------------------------------------------------------ |
| `POST` | `/predict`  | Takes `{"records": [{...}, ...]}`; returns predictions and probabilities |
| `GET`  | `/metadata` | The model's name, task, schema, feature kinds and test metrics           |
| `GET`  | `/health`   | `200` once the model is loaded; use it for liveness and readiness       |
| `GET`  | `/docs`     | Interactive OpenAPI documentation                                        |

Each record must have every feature. `null` is allowed and is imputed like a
missing value in training. Requests are validated against the schema before
they reach the model, and every problem is reported at once with a `422`:

```json
{
  "detail": [
    { "record": 0, "field": "tenure", "error": "unknown field" },
    { "record": 0, "field": "tenure_months", "error": "missing field" },
    { "record": 1, "field": "monthly_charges", "error": "expected a number or null" }
  ]
}
```

Unknown fields are rejected rather than ignored, so a misspelt column name
can't silently turn into a missing value. Requests with more records than
`MAX_BATCH_SIZE`, or a body over 5 MB, get a `413`.

## Configuration file

A complete config with every option. Only `name`, `task`, `data.source`,
`data.target` and `model.type` are required; the values shown are the
defaults.

```yaml
name: churn                   # letters, digits, ".", "_" and "-"
task: classification          # or regression
seed: 42                      # for the split, folds, search and model
n_jobs: null                  # parallel jobs for search and CV; -1 for all cores

data:
  source: data/churn.csv      # .csv, .tsv, .parquet, .json, .jsonl, or sklearn:<name>
  target: churned
  features: null              # a list to use only these columns
  drop: []                    # columns to leave out, such as IDs
  numeric: []                 # force columns numeric
  categorical: []             # force columns categorical
  test_size: 0.2
  stratify: true
  positive_class: null        # binary only; default is the last class in sorted order

preprocessing:
  numeric:
    impute: median            # median, mean or zero
    scale: standard           # standard, minmax, robust or none
  categorical:
    impute: constant          # constant (a "__missing__" category) or most_frequent
    encode: onehot            # onehot or ordinal
    max_categories: 20        # one-hot only; the rest are grouped as infrequent
    min_frequency: null       # one-hot only; a count, or a fraction of rows

model:
  type: random_forest         # a built-in name, a registered name, or package.module:Class
  params: {}                  # passed to the model
  search:                     # optional
    method: grid              # grid or random
    params:                   # the model's parameters and the values to try
      max_depth: [4, 8, null]
    cv: 5
    n_iter: 20                # random search only
    scoring: null             # any scikit-learn scorer; default f1_macro or neg_root_mean_squared_error

evaluation:
  cv_folds: 5                 # 0 turns cross-validation off
  baseline: true

output:
  dir: artifacts/<name>
```

Relative paths are relative to the directory you run `mlpipeline` from.
Search parameters belong to the model, so `max_depth` means the model's
`max_depth`. To search over preprocessing, use the full pipeline path, such
as
`preprocess__columns__numeric__impute__strategy: [mean, median]`.

Override any value from the command line without editing the file. Values
are parsed as YAML:

```bash
mlpipeline run examples/churn.yml --set seed=7 --set model.search=null --set data.drop=[customer_id,senior_citizen]
```

## Testing

```bash
make test       # all tests
make coverage   # and a coverage report
```

- **Unit tests** cover config validation and its error messages, schema
  inference and request validation, each preprocessing option, the model
  registry, and metric values against hand-worked examples.
- **Integration tests** run whole pipelines on binary, multiclass and
  regression data, and check that a saved bundle loads and predicts exactly
  what the in-memory model did, that runs with the same seed give the same
  metrics, that a tampered model file is refused, and that saving never
  replaces a directory that isn't a bundle.
- **API tests** use FastAPI's test client to check predictions,
  probabilities, unseen categories, every validation error and the size
  limits.
- **CLI tests** run each command and check its exit code.

CI runs formatting, lint, strict type checking and the tests on each push
and pull request, against the minimum Python version and the latest
release. It also trains every example, builds the Docker image, and runs
the Compose demo until the API reports healthy.

## Environment variables

`mlpipeline serve` reads each setting from a flag, or from the environment
variable shown. Flags win. Run `mlpipeline serve -h` for the full list.

| Flag               | Variable         | Default     | Purpose                                                 |
| ------------------ | ---------------- | ----------- | ------------------------------------------------------- |
| `BUNDLE` argument  | `MODEL_DIR`      |             | The bundle directory to serve                           |
| `--host`           | `HOST`           | `127.0.0.1` | Listen address; the Docker image sets `0.0.0.0`         |
| `--port`           | `PORT`           | `8000`      | Listen port                                             |
| `--max-batch-size` | `MAX_BATCH_SIZE` | `1000`      | Most records accepted in one request                    |
| `--access-log`     | `ACCESS_LOG`     | on          | Log each request                                        |
| `--log-level`      | `LOG_LEVEL`      | `info`      | `debug`, `info`, `warning` or `error` (every command)   |
| `--log-format`     | `LOG_FORMAT`     | `text`      | `text` or `json` (every command); the image sets `json` |

## Deployment

### Docker

The `Dockerfile` builds a slim Python image with the toolkit installed. It
runs as a non-root user, with a health check on `/health`. Its entrypoint is
`mlpipeline`, so the same image trains and serves:

```bash
docker build -t ml-pipeline-toolkit .

# Train into a volume...
docker run --rm -v models:/models ml-pipeline-toolkit \
  run examples/churn.yml --output-dir /models/churn

# ...then serve from it, read-only.
docker run -p 8000:8000 -v models:/models:ro -e MODEL_DIR=/models/churn ml-pipeline-toolkit
```

To train on your own data, mount it (and your config) into the container, or
build an image `FROM ml-pipeline-toolkit` that copies them in. A common
pattern is to train in CI or a scheduled job, store the bundle in object
storage, and have the serving container download it before it starts.

### Before going to production

- **Only load bundles you trust.** `model.joblib` is a Python pickle, and
  loading a pickle can run arbitrary code. The checksum catches corruption,
  not tampering by someone who can also rewrite `metadata.json`. Keep
  bundles in storage only your pipeline can write to.
- **The API has no authentication.** Put it on a private network, or behind
  a gateway or proxy that checks credentials and terminates TLS.
- **Pin the versions you serve with.** Serve with the same scikit-learn
  version you trained with. The toolkit warns about a mismatch, and
  `uv.lock` pins every version for you if you deploy from this repository.
- **Check the model, not just the code.** Look at `report.md` before
  deploying: a model that barely beats its baseline isn't ready. In CI, fail
  the build if a metric in `mlpipeline run --json` drops below a threshold.
- **Watch for drift.** A model's accuracy falls as the world changes. Run
  `mlpipeline evaluate` on recent outcomes regularly, and retrain when the
  metrics drop.
- **Scale out, not up.** Each server process holds one copy of the model and
  predicts in a thread pool. Run more containers behind a load balancer for
  more throughput.
- **Use JSON logs** (`LOG_FORMAT=json`, the image's default) so your log
  pipeline can parse them.

## Security

What the project does by default:

- Configs are read with `yaml.safe_load`, which can't run code.
- Prediction requests are validated field by field against the schema, and
  unknown fields are rejected.
- Request batches and bodies are size-limited, and the `/metadata` endpoint
  leaves out the training config and data paths.
- Bundles are checked against their SHA-256 checksum before being unpickled,
  and are never written over a directory that isn't a bundle.
- The server listens on `127.0.0.1` unless you choose otherwise.
- The Docker image runs as a non-root user and contains no build tools.
- Dependabot keeps the Python dependencies, base images and actions up to
  date.

See [before going to production](#before-going-to-production) for what you
need to add, in particular authentication on the API and trusted storage for
bundles.

Found a vulnerability? See [SECURITY.md](SECURITY.md).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md), and
[CHANGELOG.md](CHANGELOG.md) for what's changed between releases.

Are you using this project as part of your own solution? [Buy me a cup of coffee to keep me tapping away...](https://www.buymeacoffee.com/tecsmith)

This took me, [Thomas EC. Smith](https://linktr.ee/tecsmith), time and effort to put together. If you're feeling particularly generous, a financial contribution of any amount would be greatly appreciated, and thank you for your consideration!

<table align="center" width="100%" style="border-collapse: collapse;">
  <tr>
    <td align="center" width="33%" style="border: none;"><b>PayPal</b></td>
    <td align="center" width="33%" style="border: none;"><b>Wise</b></td>
    <td align="center" width="34%" style="border: none;"><b>Revolut</b></td>
  </tr>
  <tr>
    <td align="center" style="border: none;"><img src="https://tecsmith.info/assets/image/codes/paypal.webp" alt="PayPal" width="100" height="100"></td>
    <td align="center" style="border: none;"><img src="https://tecsmith.info/assets/image/codes/wise.webp" alt="Wise" width="100" height="100"></td>
    <td align="center" style="border: none;"><img src="https://tecsmith.info/assets/image/codes/revolut.webp" alt="Revolut" width="100" height="100"></td>
  </tr>
</table>

For all enquiries, [please visit my website](https://www.tecsmith.info).

## License

[MIT](LICENSE) 2026 © Thomas EC. Smith
