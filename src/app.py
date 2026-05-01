import json
import sys
from typing import cast

from llm_sdk import Small_LLM_Model

from .cli.args import parse_args
from .decoder.models import FunctionDefinition
from .decoder.constrained_decoder import ConstrainedDecoder
from .io.loader import DatasetFileLoader
from .io.writer import output_results
from .models.function_call import FunctionCallResult
from .models.validation import validate_function_payload


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


def main() -> int:
    try:
        paths = parse_args(sys.argv[1:])
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    loader = DatasetFileLoader(paths=paths)

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
        output_results(paths.output_file, generated_results)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    return 0
