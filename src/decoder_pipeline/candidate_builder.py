import json
import re

from .models import FunctionDefinition, ParameterDefinition
from .types import (
    OutputCandidate,
    OutputCandidates,
    ParameterValue,
    ParameterValues,
    ParameterValueSpace,
)


class CandidateBuilder:
    """Builds compact JSON function-call candidates from function schemas."""

    def _default_parameter_value(self, parameter_type: str) -> object:
        if parameter_type == "string":
            return ""
        if parameter_type == "number":
            return 0
        raise RuntimeError(
            "unsupported parameter type for constrained decoding: "
            f"{parameter_type}"
        )

    def extract_string_candidates(self, prompt: str) -> list[str]:
        candidates: list[str] = []

        # Extract quoted strings (both single and double quotes)
        quoted_matches = re.findall(r'"([^"\\]+)"|\'([^\'\\]+)\'', prompt)
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = value.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        # Extract bare words (fallback when no quotes)
        # Common stopwords to filter out
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "as",
            "if",
        }

        # Extract words using word boundary regex
        words = re.findall(r"\b\w+(?:['-]\w+)*\b", prompt)
        for word in words:
            # Skip stopwords and keep unique candidates
            if word.lower() not in stopwords and word not in candidates:
                candidates.append(word)

        # Fallback: if no candidates at all, return empty string
        if not candidates:
            candidates.append("")

        return candidates

    def extract_number_candidates(
        self, prompt: str
    ) -> list[float]:  # noqa: C901
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
                integer_value = (
                    0
                    if integer_value_raw is None
                    else int(abs(integer_value_raw))
                )
                fractional_digits: list[str] = []
                for tok in fractional_tokens:
                    if tok == "and":
                        continue
                    if tok not in units or units[tok] > 9:
                        return None
                    fractional_digits.append(str(units[tok]))
                if not fractional_digits:
                    return None
                fractional_part = float("0." + "".join(fractional_digits))
                return sign * (integer_value + fractional_part)
            total = 0
            current = 0
            consumed_numeric_token = False
            for tok in tokens:
                if tok == "and":
                    continue
                if tok in units:
                    current += units[tok]
                    consumed_numeric_token = True
                elif tok in tens:
                    current += tens[tok]
                    consumed_numeric_token = True
                elif tok == "hundred":
                    current = max(1, current) * 100
                    consumed_numeric_token = True
                elif tok in {"thousand", "million", "billion", "trillion"}:
                    total += max(1, current) * scales[tok]
                    current = 0
                    consumed_numeric_token = True
                else:
                    return None
            if not consumed_numeric_token:
                return None
            return sign * (total + current)

        mentions: list[tuple[int, int | float]] = []

        for match in re.finditer(r"-?\d+(?:\.\d+)?", prompt):
            numeric_value: int | float = float(match.group(0))
            if isinstance(numeric_value, float) and numeric_value.is_integer():
                numeric_value = int(numeric_value)
            mentions.append((match.start(), numeric_value))

        word_tokens_with_positions: list[tuple[str, int]] = []
        for word_match in re.finditer(r"[a-zA-Z]+", prompt.lower()):
            token = word_match.group(0)
            token_start = word_match.start()
            if token in valid_word_tokens:
                word_tokens_with_positions.append((token, token_start))
            else:
                split_parts = split_compound_number_token(token)
                if split_parts is None:
                    word_tokens_with_positions.append((token, token_start))
                else:
                    for split_part in split_parts:
                        word_tokens_with_positions.append(
                            (split_part, token_start)
                        )

        normalized_word_tokens = [tok for tok, _ in word_tokens_with_positions]
        token_positions = [pos for _, pos in word_tokens_with_positions]

        token_index = 0
        while token_index < len(normalized_word_tokens):
            if normalized_word_tokens[token_index] not in valid_word_tokens:
                token_index += 1
                continue
            best_end: int | None = None
            best_value: int | float | None = None
            for end in range(len(normalized_word_tokens), token_index, -1):
                token_slice = normalized_word_tokens[token_index:end]
                if any(tok not in valid_word_tokens for tok in token_slice):
                    continue
                parsed_value = parse_word_number(token_slice)
                if parsed_value is None:
                    continue
                if (
                    isinstance(parsed_value, float)
                    and parsed_value.is_integer()
                ):
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
        ordered_values = [float(value) for _, value in mentions]

        if not ordered_values:
            return [0.0]
        return ordered_values

    def parameter_candidates(
        self,
        prompt: str,
        parameter_definition: ParameterDefinition,
    ) -> list[ParameterValue]:
        if parameter_definition.type == "string":
            return list(self.extract_string_candidates(prompt))
        if parameter_definition.type == "number":
            return list(self.extract_number_candidates(prompt))
        return [self._default_parameter_value(parameter_definition.type)]

    def materialize_candidate_json(
        self,
        function_name: str,
        parameters: ParameterValues,
    ) -> OutputCandidate:
        return json.dumps(
            {
                "name": function_name,
                "parameters": parameters,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def expand_function_candidates_for_prompt(
        self,
        function_definition: FunctionDefinition,
        prompt: str,
        max_candidates_per_function: int = 16,
    ) -> OutputCandidates:
        parameter_names = list(function_definition.parameters.keys())
        value_space: ParameterValueSpace = {}
        for name in parameter_names:
            definition = function_definition.parameters[name]
            values = self.parameter_candidates(prompt, definition)
            if not values:
                values = [self._default_parameter_value(definition.type)]
            value_space[name] = values

        expanded: list[ParameterValues] = [{}]
        for parameter_name in parameter_names:
            next_expanded: list[ParameterValues] = []
            for partial in expanded:
                for value in value_space[parameter_name]:
                    merged = dict(partial)
                    merged[parameter_name] = value
                    next_expanded.append(merged)

            expanded = next_expanded[:max_candidates_per_function]

        if not expanded:
            fallback_parameters: ParameterValues = {}
            for name in parameter_names:
                fallback_parameters[name] = self._default_parameter_value(
                    function_definition.parameters[name].type
                )
            expanded = [fallback_parameters]

        candidate_texts: OutputCandidates = []
        seen: set[str] = set()
        for parameters in expanded:
            candidate = self.materialize_candidate_json(
                function_name=function_definition.name,
                parameters=parameters,
            )
            if candidate in seen:
                continue
            seen.add(candidate)
            candidate_texts.append(candidate)

        return candidate_texts

    def build_prompt_candidates(
        self,
        available_functions: list[FunctionDefinition],
        prompt: str,
        max_candidates_per_function: int = 16,
    ) -> OutputCandidates:
        all_candidates: OutputCandidates = []
        for function_definition in available_functions:
            all_candidates.extend(
                self.expand_function_candidates_for_prompt(
                    function_definition=function_definition,
                    prompt=prompt,
                    max_candidates_per_function=max_candidates_per_function,
                )
            )
        return all_candidates
