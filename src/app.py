import json
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from .constrain_decoder import ConstrainedDecoder, FunctionDefinition


class PromptCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class AppPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_definitions_file: Path
    prompts_file: Path
    output_file: Path


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
    if not isinstance(payload, list):
        raise RuntimeError(f"functions file must contain a JSON array: {path}")
    try:
        return [FunctionDefinition.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise RuntimeError(
            f"invalid function definition in {path}: {exc}"
        ) from exc


def load_prompts(path: Path) -> list[PromptCase]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise RuntimeError(f"prompt file must contain a JSON array: {path}")
    try:
        return [PromptCase.model_validate(item) for item in payload]
    except ValidationError as exc:
        raise RuntimeError(f"invalid prompt entry in {path}: {exc}") from exc


def summarize_dataset(
    functions: list[FunctionDefinition], prompts: list[PromptCase]
) -> DatasetSummary:
    prompt_lengths = np.array(
        [len(item.prompt) for item in prompts], dtype=np.float64
    )
    average_length = (
        float(prompt_lengths.mean()) if prompt_lengths.size else 0.0
    )
    return DatasetSummary(
        function_count=len(functions),
        prompt_count=len(prompts),
        average_prompt_length=average_length,
    )


def build_paths(argv: list[str]) -> AppPaths:
    if len(argv) != 6:
        raise RuntimeError(
            "usage: python -m src --function_definitions <path> "
            "--input <path> --output <path>"
        )

    arguments = dict(zip(argv[::2], argv[1::2], strict=True))
    function_definitions_value = (
        arguments.get("--function_definitions")
        or arguments.get("--functions_definition")
        or arguments.get("--function_defintions")
    )
    if function_definitions_value is None:
        raise RuntimeError(
            "missing required argument: --function_definitions"
        )

    try:
        return AppPaths(
            function_definitions_file=Path(function_definitions_value),
            prompts_file=Path(arguments["--input"]),
            output_file=Path(arguments["--output"]),
        )
    except KeyError as exc:
        raise RuntimeError(
            f"missing required argument: {exc.args[0]}"
        ) from None


def main() -> int:
    try:
        paths = build_paths(sys.argv[1:])
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    try:
        functions = load_functions(paths.function_definitions_file)
        prompts = load_prompts(paths.prompts_file)
        summary = summarize_dataset(functions, prompts)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    llm = Small_LLM_Model()
    decoder = ConstrainedDecoder(available_functions=functions, llm=llm)
    encoded = llm.encode("hello")
    logits = llm.get_logits_from_input_ids(encoded[0].tolist())
    returned_text = llm.decode([[24342]])
    path_to_merges_file = llm.get_path_to_merges_file()
    path_to_tokenizer_file = llm.get_path_to_tokenizer_file()
    path_to_vocab_file = llm.get_path_to_vocab_file()
    print(f"Encoded: {encoded}")
    print(f"Logits: {logits[:5]}...")
    print(f"Decoded: {returned_text}")
    print(f"Path to merges file: {path_to_merges_file}")
    print(f"Path to tokenizer file: {path_to_tokenizer_file}")
    print(f"Path to vocab file: {path_to_vocab_file}")
    print(
        "Function constraints encoded: "
        f"{len(decoder.encoded_function_definitions)}"
    )

    print("call-me-maybe scaffold")
    print(f"Functions loaded: {summary.function_count}")
    print(f"Prompts loaded: {summary.prompt_count}")
    print(f"Average prompt length: {summary.average_prompt_length:.2f}")
    return 0
