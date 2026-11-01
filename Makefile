UV ?= uv
RUN := $(UV) run
CONFIG ?= examples/churn.yml
BUNDLE ?= artifacts/churn

.PHONY: help install lint fmt fmt-check typecheck test coverage validate train serve demo data docker clean

help: ## List the available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install: ## Create .venv with the toolkit, the serve extra and dev tools
	$(UV) sync --extra serve

lint: ## Run ruff's linter
	$(RUN) ruff check .

fmt: ## Format and fix imports in every Python file
	$(RUN) ruff format .
	$(RUN) ruff check --fix --select I .

fmt-check: ## Fail if any Python file needs formatting
	$(RUN) ruff format --check .

typecheck: ## Run mypy in strict mode
	$(RUN) mypy

test: ## Run the tests
	$(RUN) pytest

coverage: ## Run the tests and write htmlcov/index.html
	$(RUN) pytest --cov --cov-report=term --cov-report=html

validate: fmt-check lint typecheck test ## Format check, lint, types and tests: run before you push

train: ## Train the example in CONFIG (default examples/churn.yml)
	$(RUN) mlpipeline run $(CONFIG)

serve: ## Serve the bundle in BUNDLE (default artifacts/churn) on :8000
	$(RUN) mlpipeline serve $(BUNDLE)

demo: ## Train the churn model and serve it with Docker Compose
	docker compose up --build

data: ## Regenerate examples/data/churn.csv
	$(RUN) python examples/make_churn_data.py

docker: ## Build the Docker image
	docker build -t ml-pipeline-toolkit .

clean: ## Remove bundles, caches and coverage output
	rm -rf artifacts htmlcov .coverage .pytest_cache .mypy_cache .ruff_cache dist
