"""Top-level orchestration for the function-calling pipeline.

Wires together the CLI argument parser, dataset loader, constrained decoder,
and output writer.  ``main()`` is the single entry point; it returns an
integer exit code so that ``__main__.py`` can forward it to the OS.

High-level data flow
--------------------
1. ``parse_args``          — resolve file paths from argv
2. ``DatasetFileLoader``   — read function definitions and prompts from disk
3. ``ConstrainedDecoder``  — for each prompt, run token-level constrained
                            decoding and emit a validated JSON function-call
4. ``output_results``      — write the list of results to the output file
"""
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
    """Build a name-keyed lookup table from a list of function definitions.

    The index is used during validation to quickly retrieve the schema for
    whichever function the decoder selected.

    Args:
        functions: Validated list of function definitions loaded from disk.

    Returns:
        A dictionary mapping each function name to its definition.

    Raises:
        RuntimeError: If two definitions share the same name, which would make
            the index ambiguous.
    """
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
    """Run the full function-calling pipeline and return an OS exit code.

    Steps
    -----
    1. Parse CLI arguments to obtain file paths.
    2. Load function definitions and test prompts from disk.
    3. Instantiate the LLM and the constrained decoder.
    4. For every prompt, run constrained decoding and validate the result.
    5. Write all results to the output JSON file.

    Returns:
        0 on success, 1 on any recoverable error (printed to stdout).
    """
    # --- Step 1: parse CLI paths -------------------------------------------
    try:
        paths = parse_args(sys.argv[1:])
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    loader = DatasetFileLoader(paths=paths)

    # --- Step 2: load dataset files from disk --------------------------------
    try:
        functions = loader.load_functions()
        prompts = loader.load_prompts()
        # Build the name → definition index once so per-prompt lookup is O(1).
        function_index = build_function_index(functions)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    # --- Step 3: initialise model and decoder --------------------------------
    llm = Small_LLM_Model()
    decoder = ConstrainedDecoder(available_functions=functions, llm=llm)

    # --- Step 4: decode each prompt ------------------------------------------
    #
    # End-to-end output assembly per prompt:
    #   prompt text
    #     -> tokenizer  -> prompt_ids (list[int])
    #     -> decoder    -> JSON string e.g. {"name":"...","parameters":{...}}
    #     -> validator  -> FunctionCallPayload (schema-checked)
    #     -> result     -> FunctionCallResult  (adds back the original prompt)
    generated_results: list[FunctionCallResult] = []
    try:
        for prompt_case in prompts:
            # Tokenise the raw prompt so the decoder can use it as the
            # rolling prefix when querying model logits.
            encoded_prompt = llm.encode(prompt_case.prompt)
            prompt_ids = cast(list[int], encoded_prompt[0].tolist())

            # Run constrained decoding: the decoder uses model logits to
            # score token choices but only allows tokens that continue one of
            # the precomputed JSON candidates.
            output = decoder.apply_decoder(
                prefix_input_ids=prompt_ids,
                prompt=prompt_case.prompt,
            )

            # Validate the decoded JSON against the selected function schema
            # before accepting it as a result.
            decoded_payload = validate_function_payload(
                json.loads(output),
                function_index,
            )
            generated_results.append(
                FunctionCallResult(
                    prompt=prompt_case.prompt,
                    name=decoded_payload.name,
                    parameters=decoded_payload.parameters,
                )
            )
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1

    # --- Step 5: write output ------------------------------------------------
    try:
        output_results(paths.output_file, generated_results)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    return 0
