# orderflow-scaling - every result of the report is rebuilt by `make all` from a clean clone.
#   make install | lint | typecheck | test | sim | data | reduce | estimate | test-hyp | report | all
SHELL := bash
UV ?= uv
OFS := $(UV) run --frozen ofs
FROM ?= 2024-11-01
TO ?= 2026-10-31
HISTORY_FROM ?= 2021-11-01
JOBS ?= 4
PROFILE ?= full
# left out of the universe by decision (pre-registration §2): gold token, wrapped bitcoin
EXCLUDE ?= PAXGUSDT WBTCUSDT

.PHONY: help install lint format typecheck test test-slow sim-check sim decide universe data \
        quality reduce estimate-dev estimate test-hyp report all docs docker up down clean

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | sort

install: ## Install the locked environment
	$(UV) sync --frozen

lint: ## ruff lint + format check
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .

format: ## Apply ruff formatting
	$(UV) run --frozen ruff format .

typecheck: ## mypy --strict on src/
	$(UV) run --frozen mypy

test: lint typecheck ## Lint, types, unit + property + simulation tests with coverage (>= 85 %)
	$(UV) run --frozen pytest --cov=ofs --cov-report=term-missing

test-slow: ## Benchmarks and long simulations (weekly CI)
	$(UV) run --frozen pytest -m slow

sim-check: ## M4 acceptance: fBm autocovariance and Hawkes branching ratio
	$(OFS) simulate check

sim: ## Simulation study of the estimators (O2): bias, RMSE, coverage
	$(OFS) simulate study --profile $(PROFILE) --jobs $(JOBS)

decide: ## Pre-registration rules (§4.1 estimator choice, §5 calibration) applied to the study
	$(OFS) simulate decide --profile $(PROFILE)

universe: ## Rank the USDT pairs and write config/universe.yaml (review, then --freeze)
	$(OFS) universe --from $(FROM) --to $(TO) $(foreach s,$(EXCLUDE),--exclude $(s))

data: ## Download + validate + reduce, raw files deleted as soon as reduced; 5-year daily RV
	$(OFS) backfill --from $(FROM) --to $(TO) --jobs $(JOBS)
	$(OFS) history --from $(HISTORY_FROM) --to $(TO)

quality: ## O1 data-quality report: the state of every pair-day of the window
	$(OFS) quality --from $(FROM) --to $(TO)

reduce: ## Re-run validation and reduction on downloaded raw files only
	$(OFS) reduce --from $(FROM) --to $(TO) --jobs $(JOBS)

estimate-dev: ## Development sample (requires the git tag prereg-v1)
	$(OFS) estimate --sample development --jobs $(JOBS)

estimate: estimate-dev ## Both samples (confirmation requires the git tag config-frozen)
	$(OFS) estimate --sample confirmation --jobs $(JOBS)
	$(OFS) estimate --sample history --jobs $(JOBS)

test-hyp: ## H1-H5, negative controls, power, specification curve
	$(OFS) test --sample confirmation

forecast: ## Forecasting extension F: development (prereg-v1), then confirmation (config-frozen)
	$(OFS) forecast --sample development --jobs $(JOBS)
	$(OFS) forecast --sample confirmation --jobs $(JOBS)

changepoints: ## Change-point monitor of H0 on 14-day blocks (exploratory, after prereg-v1)
	$(OFS) changepoints --to $(TO)

sim-forecast: ## Forecast models on simulated rough volatility (QLIKE relative to HAR)
	$(OFS) simulate forecast --reps 10 --jobs $(JOBS)

report: ## Figures, LaTeX tables and paper/main.pdf
	$(OFS) report --sample confirmation --profile $(PROFILE)

all: data sim estimate test-hyp sim-forecast forecast report ## Everything, from a clean clone

docs: ## Documentation site (MkDocs)
	$(UV) run --frozen mkdocs build --strict

docker: ## Build the production image
	docker build --build-arg GIT_SHA=$$(git rev-parse --short=12 HEAD) -t orderflow-scaling:local .

up: ## Start pipeline + api + dashboard + caddy
	docker compose up -d

down:
	docker compose down

clean: ## Remove caches and generated report artefacts (never data/)
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis .benchmarks htmlcov coverage.xml site
	rm -f paper/figures/*.pdf paper/tables/*.tex
