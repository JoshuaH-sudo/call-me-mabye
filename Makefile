PYTHON := uv run python

.PHONY: install run debug clean lint lint-strict

install:
	uv sync --dev

run:
	$(PYTHON) -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

debug:
	$(PYTHON) -m pdb src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calls.json

clean:
	find . \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) -type d -prune -exec rm -rf {} +
	find . \( -name '*.pyc' -o -name '*.pyo' \) -type f -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict