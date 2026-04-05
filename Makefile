PYTHON := uv run python

.PHONY: install run debug clean lint lint-strict

install:
	uv sync --dev

run:
	$(PYTHON) -m call_me_maybe

debug:
	$(PYTHON) -m pdb -m call_me_maybe

clean:
	find . \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) -type d -prune -exec rm -rf {} +
	find . \( -name '*.pyc' -o -name '*.pyo' \) -type f -delete

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict