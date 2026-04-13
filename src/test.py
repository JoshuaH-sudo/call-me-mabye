import json

import llm_sdk

example_function_definition = [
    {
        "name": "fn_greet",
        "description": "Generate a greeting message for a person by name.",
        "parameters": {"name": {"type": "string"}},
        "returns": {"type": "string"},
    },
    {
        "name": "fn_subtract_numbers",
        "description": (
            "Subtract one number from another and return the result."
        ),
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    },
    {
        "name": "fn_count_characters",
        "description": "Count the number of characters in a string.",
        "parameters": {"s": {"type": "string"}},
        "returns": {"type": "number"},
    },
]


def test() -> None:
    llm = llm_sdk.Small_LLM_Model()
    prompt = "say hello to josh"
    # Step 0: Encode the available function definitions as model context so the
    # descriptions can influence token scores during constrained decoding.
    function_context = json.dumps(
        example_function_definition,
        separators=(",", ":"),
        sort_keys=True,
    )
    function_context_token_ids = llm.encode(function_context)[0].tolist()
    prompt_token_ids = llm.encode(prompt)[0].tolist()

    # Step 1: Convert each function definition into one valid function-call
    # candidate. The decoder will only be allowed to emit one of these compact
    # JSON objects.
    encoded_function_calls: list[list[int]] = []
    for definition in example_function_definition:
        parameters: dict[str, object] = {}
        for parameter_name, parameter_definition in definition[
            "parameters"
        ].items():
            parameter_type = parameter_definition["type"]
            if parameter_type == "string":
                parameters[parameter_name] = ""
            elif parameter_type == "number":
                parameters[parameter_name] = 0
            else:
                raise RuntimeError(
                    f"unsupported parameter type: {parameter_type}"
                )

        candidate_text = json.dumps(
            {
                "name": definition["name"],
                "parameters": parameters,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        encoded = llm.encode(
            candidate_text,
        )[0].tolist()
        print(f"Candidate: {candidate_text}")
        print(f"Encoded function-call token ids: {encoded}")
        encoded_function_calls.append(encoded)

    print(f"Function context token ids: {function_context_token_ids}")
    print(f"Prompt token ids: {prompt_token_ids}")

    # Step 2: Keep the generated output separate from the prompt. The prompt
    # remains model context only; the constraint logic operates on the output
    # tokens we have generated so far.
    generated_token_ids: list[int] = []
    max_new_tokens = 50
    stop_reason = "max_new_tokens was reached."

    for _ in range(max_new_tokens):
        # Step 3: Collect the only next tokens that still keep the generated
        # output as a prefix of at least one valid function-call candidate.
        allowed_token_ids: set[int] = set()
        for encoded_function_call in encoded_function_calls:
            prefix_length = len(generated_token_ids)
            # if the generated output is already longer than this candidate,
            # it can't be a valid continuation
            if prefix_length >= len(encoded_function_call):
                continue
            # if the generated output doesn't match the start of this
            # candidate, it can't be a valid continuation
            if encoded_function_call[:prefix_length] != generated_token_ids:
                continue
            # else this candidate is still valid, so add the next token in this
            # candidate to the allowed set
            allowed_token_ids.add(encoded_function_call[prefix_length])

        if not allowed_token_ids:
            stop_reason = "no valid constrained continuation remained."
            break

        # Step 4: Score the next token using function definitions + prompt +
        # generated output, then pick the best token only from the allowed set.
        # The mask still enforces correctness; the extra context only helps the
        # model rank which constrained branch fits the prompt better.
        rolling_prefix = (
            function_context_token_ids + prompt_token_ids + generated_token_ids
        )
        logits = llm.get_logits_from_input_ids(rolling_prefix)
        next_token_id = max(
            allowed_token_ids,
            key=lambda token_id: logits[token_id],
        )

        # Step 5: Append the chosen token to the generated output.
        generated_token_ids.append(next_token_id)
        print(
            f"Generated token {next_token_id}: {llm.decode([next_token_id])!r}"
        )

        # Step 6: Stop as soon as the generated output exactly matches one full
        # valid function-call candidate.
        if any(
            generated_token_ids == encoded_function_call
            for encoded_function_call in encoded_function_calls
        ):
            stop_reason = "a complete constrained candidate was generated."
            break

    print(f"Stopping because {stop_reason}")

    print(f"Generated token ids: {generated_token_ids}")
    print(f"prompt: {prompt!r}")
    print(f"Completion: {llm.decode(generated_token_ids)!r}\n\n")


if __name__ == "__main__":
    test()
