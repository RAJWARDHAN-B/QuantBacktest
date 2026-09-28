T ?= MSFT
COMPOSE := docker compose

.PHONY: build test lint fmt analyze fetch shell

build:
	$(COMPOSE) build

test:
	$(COMPOSE) run --rm tests

lint:
	$(COMPOSE) run --rm tests sh -c "ruff check src tests && ruff format --check src tests && mypy"

fmt:
	$(COMPOSE) run --rm tests sh -c "ruff check --fix src tests && ruff format src tests"

analyze:
	$(COMPOSE) run --rm analyst analyze $(T)

fetch:
	$(COMPOSE) run --rm analyst fetch $(T)

shell:
	$(COMPOSE) run --rm --entrypoint bash analyst
