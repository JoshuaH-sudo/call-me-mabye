from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class ReturnDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class PromptCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class AppPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    functions_file: Path
    prompts_file: Path


class DatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_count: int
    prompt_count: int
    average_prompt_length: float


def _load_json(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise RuntimeError(f"missing input file: {path}") from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc.msg}") from exc
    except OSError as exc:
        raise RuntimeError(f"unable to read {path}: {exc}") from exc


def load_functions(path: Path) -> list[FunctionDefinition]:
    payload = _load_json(path)
    try:
        return [FunctionDefinition.model_validate(item) for item in payload]
    except TypeError as exc:
        raise RuntimeError(f"functions file must contain a JSON array: {path}") from exc
    except ValidationError as exc:
        raise RuntimeError(f"invalid function definition in {path}: {exc}") from exc


def load_prompts(path: Path) -> list[PromptCase]:
    payload = _load_json(path)
    try:
        return [PromptCase.model_validate(item) for item in payload]
    except TypeError as exc:
        raise RuntimeError(f"prompt file must contain a JSON array: {path}") from exc
    except ValidationError as exc:
        raise RuntimeError(f"invalid prompt entry in {path}: {exc}") from exc


def summarize_dataset(
    functions: list[FunctionDefinition], prompts: list[PromptCase]
) -> DatasetSummary:
    prompt_lengths = np.array([len(item.prompt) for item in prompts], dtype=np.float64)
    average_length = float(prompt_lengths.mean()) if prompt_lengths.size else 0.0
    return DatasetSummary(
        function_count=len(functions),
        prompt_count=len(prompts),
        average_prompt_length=average_length,
    )


def build_paths(project_root: Path) -> AppPaths:
    return AppPaths(
        functions_file=project_root / "data" / "input" / "functions_definition.json",
        prompts_file=project_root / "data" / "input" / "function_calling_tests.json",
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    paths = build_paths(project_root)

    try:
        functions = load_functions(paths.functions_file)
        prompts = load_prompts(paths.prompts_file)
        summary = summarize_dataset(functions, prompts)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    print("call-me-maybe scaffold")
    print(f"Functions loaded: {summary.function_count}")
    print(f"Prompts loaded: {summary.prompt_count}")
    print(f"Average prompt length: {summary.average_prompt_length:.2f}")
    return 0