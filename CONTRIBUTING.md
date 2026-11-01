# Contributing

Thanks for helping improve the toolkit. Bug reports, fixes and focused
improvements are all welcome.

## Getting set up

Follow the [quick start](README.md#quick-start) to install it and train the
example, then check everything passes:

```bash
make validate   # format check, lint, strict type check, tests
```

## Making a change

1. Open an issue first for anything larger than a small fix, so we can agree
   on the approach before you put time in.
2. Branch from `main` and keep the change focused on one thing.
3. Add or update tests for behaviour you change.
4. Add a line under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for any
   change users will notice.
5. Run `make fmt` and `make validate` before pushing.
6. Open a pull request describing what changed and how you tested it.

## Guidelines

- Keep the core small. New dependencies go in an optional extra (like
  `serve` and `parquet`) unless every user needs them. Prefer an extension
  point, such as `register_model` or an import path, and a note in the
  README over adding a library.
- Build on scikit-learn's interfaces rather than wrapping them. Anything that
  transforms data belongs in the saved pipeline, so training and serving
  can't drift apart.
- Config errors should name the exact key and say what was expected. Add a
  case to `tests/test_config.py` for each new validation rule.
- A change to `metadata.json` that older versions can't read needs a new
  `BUNDLE_FORMAT` in `artifacts.py` and a CHANGELOG entry.
- Tests must be deterministic: pass a seed, and use small in-memory data
  rather than downloads.
- Public functions need type hints and a docstring; `mypy --strict` runs in
  CI.
- Every source file ends with the project's copyright footer.
