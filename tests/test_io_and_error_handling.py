"""Tests for dataset file loading and graceful error handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app import main
from src.cli.paths import AppPaths
from src.decoder.models import FunctionDefinition
from src.io.loader import DatasetFileLoader


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_function_definition_payload() -> list[dict[str, object]]:
    return [
        {
            "name": "add",
            "description": "Adds two numbers.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "returns": {"type": "number"},
        }
    ]


def _valid_prompt_payload() -> list[dict[str, str]]:
    return [{"prompt": "What is 2 plus 3?"}]


def _make_loader(
    tmp_path: Path,
    functions_payload: object,
    prompts_payload: object,
) -> DatasetFileLoader:
    functions_file = tmp_path / "function_definitions.json"
    prompts_file = tmp_path / "function_calling_tests.json"
    output_file = tmp_path / "function_calls.json"

    _write_json(functions_file, functions_payload)
    _write_json(prompts_file, prompts_payload)

    return DatasetFileLoader(
        paths=AppPaths(
            function_definitions_file=functions_file,
            prompts_file=prompts_file,
            output_file=output_file,
        )
    )


def test_load_prompts_reads_function_calling_tests_json(tmp_path: Path) -> None:
    loader = _make_loader(
        tmp_path=tmp_path,
        functions_payload=_valid_function_definition_payload(),
        prompts_payload=_valid_prompt_payload(),
    )

    prompts = loader.load_prompts()

    assert len(prompts) == 1
    assert prompts[0].prompt == "What is 2 plus 3?"


def test_load_functions_reads_function_definitions_json(tmp_path: Path) -> None:
    loader = _make_loader(
        tmp_path=tmp_path,
        functions_payload=_valid_function_definition_payload(),
        prompts_payload=_valid_prompt_payload(),
    )

    functions = loader.load_functions()

    assert len(functions) == 1
    assert isinstance(functions[0], FunctionDefinition)
    assert functions[0].name == "add"


def test_invalid_json_in_prompts_file_raises_runtime_error(tmp_path: Path) -> None:
    functions_file = tmp_path / "function_definitions.json"
    prompts_file = tmp_path / "function_calling_tests.json"
    output_file = tmp_path / "function_calls.json"

    _write_json(functions_file, _valid_function_definition_payload())
    prompts_file.write_text("{ invalid json", encoding="utf-8")

    loader = DatasetFileLoader(
        paths=AppPaths(
            function_definitions_file=functions_file,
            prompts_file=prompts_file,
            output_file=output_file,
        )
    )

    with pytest.raises(RuntimeError, match=r"invalid JSON in"):
        loader.load_prompts()


def test_missing_function_definitions_file_raises_clear_error(
    tmp_path: Path,
) -> None:
    prompts_file = tmp_path / "function_calling_tests.json"
    _write_json(prompts_file, _valid_prompt_payload())

    loader = DatasetFileLoader(
        paths=AppPaths(
            function_definitions_file=tmp_path / "missing_functions.json",
            prompts_file=prompts_file,
            output_file=tmp_path / "function_calls.json",
        )
    )

    with pytest.raises(RuntimeError, match=r"missing input file:"):
        loader.load_functions()


def test_main_handles_missing_input_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    missing_paths = AppPaths(
        function_definitions_file=tmp_path / "missing_functions.json",
        prompts_file=tmp_path / "missing_prompts.json",
        output_file=tmp_path / "function_calls.json",
    )

    monkeypatch.setattr("src.app.parse_args", lambda _argv: missing_paths)

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: missing input file:" in captured.out
