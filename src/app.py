import json
import sys
from pathlib import Path
from typing import cast

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict
from .constrain_decoder import ConstrainedDecoder, FunctionDefinition
from .data_loader import DatasetFileLoader


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


class FunctionCallPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, object]


def build_function_index(
    functions: list[FunctionDefinition],
) -> dict[str, FunctionDefinition]:
    function_index: dict[str, FunctionDefinition] = {}
    for function_definition in functions:
        if function_definition.name in function_index:
            raise RuntimeError(
                "duplicate function name in definitions: "
                f"{function_definition.name}"
            )
        function_index[function_definition.name] = function_definition
    return function_index


def is_valid_parameter_value(
    parameter_type: str,
    value: object,
) -> bool:
    if parameter_type == "string":
        return isinstance(value, str)
    if parameter_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise RuntimeError(f"unsupported parameter type: {parameter_type}")


def validate_function_payload(
    payload: object,
    function_index: dict[str, FunctionDefinition],
) -> FunctionCallPayload:
    validated_payload = FunctionCallPayload.model_validate(payload)

    function_definition = function_index.get(validated_payload.name)
    if function_definition is None:
        raise RuntimeError(
            "decoder selected an unknown function name: "
            f"{validated_payload.name}"
        )

    expected_parameter_names = set(function_definition.parameters)
    actual_parameter_names = set(validated_payload.parameters)
    if actual_parameter_names != expected_parameter_names:
        missing_names = sorted(
            expected_parameter_names - actual_parameter_names
        )
        extra_names = sorted(actual_parameter_names - expected_parameter_names)
        raise RuntimeError(
            "decoded parameters do not match function schema for "
            f"{validated_payload.name}: missing={missing_names}, "
            f"extra={extra_names}"
        )

    for (
        parameter_name,
        parameter_definition,
    ) in function_definition.parameters.items():
        value = validated_payload.parameters[parameter_name]
        if not is_valid_parameter_value(parameter_definition.type, value):
            raise RuntimeError(
                "decoded parameter has the wrong type for "
                f"{validated_payload.name}.{parameter_name}: expected "
                f"{parameter_definition.type}, got {type(value).__name__}"
            )

    return validated_payload


def output_results(
    output_file: Path,
    results: list[FunctionCallResult],
) -> None:
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(
                [item.model_dump(mode="json") for item in results],
                handle,
                ensure_ascii=False,
                indent=2,
            )
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
        function_index = build_function_index(functions)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    llm = Small_LLM_Model()
    decoder = ConstrainedDecoder(available_functions=functions, llm=llm)

    # First milestone: emit schema-shaped JSON strings chosen under token
    # constraints. Parameter values are placeholder defaults.
    #
    # End-to-end output assembly:
    # prompt text
    #   -> model prompt ids
    #   -> constrained decoder emits JSON like:
    #      {"name":"...","parameters":{...}}
    #   -> app wraps it into the final result object:
    #      {"prompt":"original prompt","name":"...","parameters":{...}}
    generated_results: list[FunctionCallResult] = []
    try:
        for prompt_case in prompts:
            # Function selection is driven by the model logits from the raw
            # user prompt while constrained decoding limits the output to the
            # valid function-call JSON candidates.
            encoded_prompt = llm.encode(prompt_case.prompt)
            prompt_ids = cast(list[int], encoded_prompt[0].tolist())
            output = decoder.apply_decoder(
                prefix_input_ids=prompt_ids,
                prompt=prompt_case.prompt,
            )
            decoded_payload = validate_function_payload(
                json.loads(output),
                function_index,
            )
            generated_results.append(
                # The decoder enforces the inner function-call payload and the
                # app validates it against the selected function schema before
                # attaching the original prompt.
                FunctionCallResult(
                    prompt=prompt_case.prompt,
                    name=decoded_payload.name,
                    parameters=decoded_payload.parameters,
                )
            )
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1

    try:
        output_results(loader.paths.output_file, generated_results)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    return 0
