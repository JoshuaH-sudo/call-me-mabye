import json
import sys
from pathlib import Path
from typing import cast

from llm_sdk import Small_LLM_Model
import numpy as np
from pydantic import BaseModel, ConfigDict
from .constrain_decoder import ConstrainedDecoder, FunctionDefinition
from .data_loader import DatasetFileLoader, PromptCase


class DatasetSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function_count: int
    prompt_count: int
    average_prompt_length: float


class FunctionCallResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]


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


def append_results(
    output_file: Path,
    results: list[FunctionCallResult],
) -> None:
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("a", encoding="utf-8") as handle:
            for item in results:
                handle.write(item.model_dump_json())
                handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"unable to write output file {output_file}: {exc}")


def main() -> int:
    try:
        loader = DatasetFileLoader.from_argv(sys.argv[1:])
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    try:
        functions = loader.load_functions()
        prompts = loader.load_prompts()
        summary = summarize_dataset(functions, prompts)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    llm = Small_LLM_Model()
    decoder = ConstrainedDecoder(available_functions=functions, llm=llm)

    # First milestone: emit schema-shaped JSON strings chosen under token
    # constraints. Parameter values are placeholder defaults.
    generated_results: list[FunctionCallResult] = []
    for prompt_case in prompts:
        encoded_prompt = llm.encode(prompt_case.prompt)
        prompt_ids = cast(list[int], encoded_prompt[0].tolist())
        output = decoder.force_json_output(
            prefix_input_ids=prompt_ids,
        )
        payload = json.loads(output)
        generated_results.append(
            FunctionCallResult(
                prompt=prompt_case.prompt,
                name=payload["name"],
                parameters=payload["parameters"],
            )
        )

    print("Generated results:")
    for item in generated_results:
        print(item)

    try:
        append_results(loader.paths.output_file, generated_results)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Appended {len(generated_results)} result line(s)")
    print(f"Output path: {loader.paths.output_file}")

    print("call-me-maybe scaffold")
    print(f"Functions loaded: {summary.function_count}")
    print(f"Prompts loaded: {summary.prompt_count}")
    print(f"Average prompt length: {summary.average_prompt_length:.2f}")
    return 0
