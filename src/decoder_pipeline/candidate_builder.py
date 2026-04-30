import json
import re

from .models import FunctionDefinition, ParameterDefinition
from .number_parameter_extractor import NumberParameterExtractor
from .regex_parameter_extractor import RegexParameterExtractor
from .string_parameter_extractor import StringParameterExtractor
from .types import (
    OutputCandidate,
    OutputCandidates,
    ParameterValue,
    ParameterValues,
    ParameterValueSpace,
)

_STOPWORDS = {
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


def _tokenize(text: str) -> set[str]:
    return {
        w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOPWORDS
    }


def _score_function(fn: FunctionDefinition, prompt_tokens: set[str]) -> int:
    name_tokens = {
        p for p in fn.name.lower().split("_") if p not in _STOPWORDS
    }
    fn_tokens = _tokenize(fn.description) | name_tokens
    score = len(fn_tokens & prompt_tokens)
    if {"replace", "substitute"} & prompt_tokens and "substitute" in fn.name:
        score += 5
    return score


class CandidateBuilder:
    """Builds compact JSON function-call candidates from function schemas.

    Coordinates parameter extraction and JSON candidate generation.
    Delegates extraction logic to specialized extractor classes.
    """

    def __init__(self) -> None:
        """Initialize the candidate builder with specialized extractors."""
        self.string_extractor = StringParameterExtractor()
        self.number_extractor = NumberParameterExtractor()
        self.regex_extractor = RegexParameterExtractor()

    def _default_parameter_value(self, parameter_type: str) -> object:
        if parameter_type == "string":
            return ""
        if parameter_type == "number":
            return 0
        raise RuntimeError(
            "unsupported parameter type for constrained decoding: "
            f"{parameter_type}"
        )

    def parameter_candidates(
        self,
        prompt: str,
        parameter_definition: ParameterDefinition,
        parameter_name: str = "",
    ) -> list[ParameterValue]:
        """Extract parameter candidates based on parameter type.

        Delegates to specialized extractors:
        - RegexParameterExtractor for parameters named "regex"
        - StringParameterExtractor for string types
        - NumberParameterExtractor for number types
        """
        if parameter_name == "regex":
            return list(self.regex_extractor.extract_candidates(prompt))
        if parameter_definition.type == "string":
            return list(
                self.string_extractor.extract_candidates(
                    prompt,
                    parameter_name,
                )
            )
        if parameter_definition.type == "number":
            return list(self.number_extractor.extract_candidates(prompt))
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
            values = self.parameter_candidates(prompt, definition, name)
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
        prompt_tokens = _tokenize(prompt)

        def has_candidates_for_all_params(fn: FunctionDefinition) -> bool:
            return all(
                bool(self.parameter_candidates(prompt, defn, name))
                for name, defn in fn.parameters.items()
            )

        sorted_fns = sorted(
            available_functions,
            key=lambda fn: _score_function(fn, prompt_tokens),
            reverse=True,
        )
        filtered_fns = [
            fn for fn in sorted_fns if has_candidates_for_all_params(fn)
        ]

        if not filtered_fns:
            filtered_fns = sorted_fns[:1]

        top_fn = filtered_fns[:1]

        all_candidates: OutputCandidates = []
        for function_definition in top_fn:
            all_candidates.extend(
                self.expand_function_candidates_for_prompt(
                    function_definition=function_definition,
                    prompt=prompt,
                    max_candidates_per_function=max_candidates_per_function,
                )
            )
        return all_candidates
