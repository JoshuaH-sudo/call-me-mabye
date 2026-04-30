import json

from .models import FunctionDefinition, ParameterDefinition
from .number_parameter_extractor import NumberParameterExtractor
from .string_parameter_extractor import StringParameterExtractor
from .types import (
    OutputCandidate,
    OutputCandidates,
    ParameterValue,
    ParameterValues,
    ParameterValueSpace,
)


class CandidateBuilder:
    """Builds compact JSON function-call candidates from function schemas.

    Coordinates parameter extraction and JSON candidate generation.
    Delegates extraction logic to specialized extractor classes.
    """

    def __init__(self) -> None:
        """Initialize the candidate builder with specialized extractors."""
        self.string_extractor = StringParameterExtractor()
        self.number_extractor = NumberParameterExtractor()

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
    ) -> list[ParameterValue]:
        """Extract parameter candidates based on parameter type.

        Delegates to specialized extractors:
        - StringParameterExtractor for string types
        - NumberParameterExtractor for number types
        """
        if parameter_definition.type == "string":
            return list(self.string_extractor.extract_candidates(prompt))
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
