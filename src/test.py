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
    prompt = "what is 10 - 5?"
    prompt_token_ids = llm.encode(prompt)[0].tolist()

    # Step 1: Build the exact output candidates we will allow the model to
    # generate. For this experiment, each full JSON function definition is one
    # valid target sequence.
    encoded_function_definitions: list[list[int]] = []
    for definition in example_function_definition:
        encoded = llm.encode(
            json.dumps(
                definition,
                separators=(",", ":"),
                sort_keys=True,
            )
        )[0].tolist()
        print(f"Encoded function definition token ids: {encoded}")
        encoded_function_definitions.append(encoded)

    print(f"Prompt token ids: {prompt_token_ids}")

    # Step 2: Keep the generated output separate from the prompt. The prompt is
    # model context only; the constraint logic must operate on generated tokens
    # alone.
    generated_token_ids: list[int] = []
    max_new_tokens = 50
    stop_reason = "max_new_tokens was reached."

    for _ in range(max_new_tokens):
        # Step 3: Find the only next tokens that keep the current generated
        # output as a prefix of at least one full candidate.
        allowed_token_ids: set[int] = set()
        for encoded_definition in encoded_function_definitions:
            prefix_length = len(generated_token_ids)
            if prefix_length >= len(encoded_definition):
                continue
            if encoded_definition[:prefix_length] != generated_token_ids:
                continue
            allowed_token_ids.add(encoded_definition[prefix_length])

        if not allowed_token_ids:
            stop_reason = "no valid constrained continuation remained."
            break

        # Step 4: Score the next token using the prompt plus everything we have
        # already generated, then mask out every token that is not allowed.
        rolling_prefix = prompt_token_ids + generated_token_ids
        logits = llm.get_logits_from_input_ids(rolling_prefix)
        next_token_id = max(
            allowed_token_ids,
            key=lambda token_id: logits[token_id],
        )

        # Step 5: Append the chosen token and stop as soon as we exactly match
        # one full allowed candidate.
        generated_token_ids.append(next_token_id)
        print(
            f"Generated token {next_token_id}: {llm.decode([next_token_id])!r}"
        )

        if any(
            generated_token_ids == encoded_definition
            for encoded_definition in encoded_function_definitions
        ):
            stop_reason = "a complete constrained candidate was generated."
            break

    print(f"Stopping because {stop_reason}")

    print(f"Generated token ids: {generated_token_ids}")
    print(f"prompt: {prompt!r}")
    print(f"Completion: {llm.decode(generated_token_ids)!r}\n\n")


if __name__ == "__main__":
    test()
