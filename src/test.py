import json
import re

import llm_sdk

example_function_definition = [
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers together and return their sum.",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    },
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


def _build_string_candidate_values(prompt: str) -> list[str]:
    words = prompt.split()
    values: list[str] = [""]

    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            values.append(" ".join(words[start:end]))

    # Keep insertion order while removing duplicates.
    return list(dict.fromkeys(values))


def _build_number_candidate_values(prompt: str) -> list[int | float]:
    units: dict[str, int] = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
    }
    tens: dict[str, int] = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    scales: dict[str, int] = {
        "hundred": 100,
        "thousand": 1_000,
        "million": 1_000_000,
        "billion": 1_000_000_000,
        "trillion": 1_000_000_000_000,
    }
    valid_word_tokens = (
        set(units)
        | set(tens)
        | set(scales)
        | {"and", "point", "minus", "negative"}
    )
    compound_number_parts = set(units) | set(tens) | set(scales)

    def split_compound_number_token(token: str) -> list[str] | None:
        memo: dict[str, list[str] | None] = {}

        def solve(rest: str) -> list[str] | None:
            if rest == "":
                return []
            if rest in memo:
                return memo[rest]

            for part in compound_number_parts:
                if rest.startswith(part):
                    tail = solve(rest[len(part) :])
                    if tail is not None:
                        memo[rest] = [part] + tail
                        return memo[rest]

            memo[rest] = None
            return None

        parts = solve(token)
        if parts is None or len(parts) <= 1:
            return None
        return parts

    def parse_word_number(tokens: list[str]) -> int | float | None:
        if not tokens:
            return None

        sign = 1
        if tokens[0] in {"minus", "negative"}:
            sign = -1
            tokens = tokens[1:]

        if not tokens:
            return None

        if "point" in tokens:
            point_index = tokens.index("point")
            integer_tokens = tokens[:point_index]
            fractional_tokens = tokens[point_index + 1 :]

            if not fractional_tokens:
                return None

            integer_value_raw = parse_word_number(integer_tokens)
            integer_value = 0
            if integer_value_raw is not None:
                integer_value = int(abs(integer_value_raw))

            fractional_digits: list[str] = []
            for token in fractional_tokens:
                if token == "and":
                    continue
                if token not in units or units[token] > 9:
                    return None
                fractional_digits.append(str(units[token]))

            if not fractional_digits:
                return None

            fractional_part = float("0." + "".join(fractional_digits))
            return sign * (integer_value + fractional_part)

        total = 0
        current = 0
        consumed_numeric_token = False

        for token in tokens:
            if token == "and":
                continue
            if token in units:
                current += units[token]
                consumed_numeric_token = True
                continue
            if token in tens:
                current += tens[token]
                consumed_numeric_token = True
                continue
            if token == "hundred":
                current = max(1, current) * 100
                consumed_numeric_token = True
                continue
            if token in {"thousand", "million", "billion", "trillion"}:
                total += max(1, current) * scales[token]
                current = 0
                consumed_numeric_token = True
                continue
            return None

        if not consumed_numeric_token:
            return None

        return sign * (total + current)

    mentions: list[tuple[int, int | float]] = []

    # Capture digit-based numbers such as -12, 1,234, 3.14, and 2e5.
    digit_number_regex = (
        r"(?<![\\w.])[+-]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?"
        r"(?:[eE][+-]?\\d+)?(?![\\w.])"
    )
    for match in re.finditer(digit_number_regex, prompt):
        normalized_match = match.group(0).replace(",", "")
        numeric_value = float(normalized_match)
        if numeric_value.is_integer():
            mentions.append((match.start(), int(numeric_value)))
        else:
            mentions.append((match.start(), numeric_value))

    word_tokens_with_positions: list[tuple[str, int]] = []
    for word_match in re.finditer(r"[a-zA-Z]+", prompt.lower()):
        token = word_match.group(0)
        token_start = word_match.start()

        if token in valid_word_tokens:
            word_tokens_with_positions.append((token, token_start))
            continue

        split_parts = split_compound_number_token(token)
        if split_parts is None:
            word_tokens_with_positions.append((token, token_start))
            continue

        for split_part in split_parts:
            word_tokens_with_positions.append((split_part, token_start))

    normalized_word_tokens = [token for token, _ in word_tokens_with_positions]
    token_positions = [position for _, position in word_tokens_with_positions]

    token_index = 0
    while token_index < len(normalized_word_tokens):
        if normalized_word_tokens[token_index] not in valid_word_tokens:
            token_index += 1
            continue

        best_end: int | None = None
        best_value: int | float | None = None

        for end in range(len(normalized_word_tokens), token_index, -1):
            token_slice = normalized_word_tokens[token_index:end]
            if any(token not in valid_word_tokens for token in token_slice):
                continue

            parsed_value = parse_word_number(token_slice)
            if parsed_value is None:
                continue

            if isinstance(parsed_value, float) and parsed_value.is_integer():
                best_value = int(parsed_value)
            else:
                best_value = parsed_value
            best_end = end
            break

        if best_end is None or best_value is None:
            token_index += 1
            continue

        mentions.append((token_positions[token_index], best_value))
        token_index = best_end

    mentions.sort(key=lambda item: item[0])
    ordered_values = [value for _, value in mentions]

    if not ordered_values:
        return [0]

    return ordered_values


def _build_parameter_candidates(
    parameter_definitions: dict[str, dict[str, str]],
    prompt: str,
) -> list[dict[str, object]]:
    parameter_items = list(parameter_definitions.items())

    if parameter_items and all(
        parameter_definition["type"] == "number"
        for _, parameter_definition in parameter_items
    ):
        ordered_numbers = _build_number_candidate_values(prompt)
        if len(ordered_numbers) >= len(parameter_items):
            ordered_mapping: dict[str, object] = {}
            for index, (parameter_name, _) in enumerate(parameter_items):
                ordered_mapping[parameter_name] = ordered_numbers[index]
            return [ordered_mapping]

    parameter_candidates: list[dict[str, object]] = [{}]

    for parameter_name, parameter_definition in parameter_items:
        parameter_type = parameter_definition["type"]

        if parameter_type == "string":
            possible_values = _build_string_candidate_values(prompt)
        elif parameter_type == "number":
            possible_values = _build_number_candidate_values(prompt)
        else:
            raise RuntimeError(f"unsupported parameter type: {parameter_type}")

        next_parameter_candidates: list[dict[str, object]] = []
        for candidate in parameter_candidates:
            for value in possible_values:
                next_candidate = dict(candidate)
                next_candidate[parameter_name] = value
                next_parameter_candidates.append(next_candidate)

        parameter_candidates = next_parameter_candidates

    return parameter_candidates


def test() -> None:
    llm = llm_sdk.Small_LLM_Model()
    prompt = "What is the sum of twentyone and 20?"
    string_candidate_values = _build_string_candidate_values(prompt)
    print(f"String candidate values: {string_candidate_values}")

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
        parameter_candidates = _build_parameter_candidates(
            definition["parameters"],
            prompt,
        )

        for parameters in parameter_candidates:
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
