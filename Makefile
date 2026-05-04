PYTHON := uv run python

.PHONY: install run debug test clean lint lint-strict

.venv:
	uv venv .venv

install: .venv
	uv sync --dev

run: .venv
	$(PYTHON) -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

debug: .venv
	$(PYTHON) -m pdb -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json \
		--debug

test: .venv
	$(PYTHON) -m pytest -q

clean:
	find . \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) -type d -prune -exec rm -rf {} +
	find . \( -name '*.pyc' -o -name '*.pyo' \) -type f -delete

lint: .venv
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict: .venv
	uv run flake8 src
	uv run mypy src --strict