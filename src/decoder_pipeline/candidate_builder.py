import json
import re

from .models import FunctionDefinition, ParameterDefinition


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

    def build_base_candidates(
        self,
        available_functions: list[FunctionDefinition],
    ) -> list[str]:
        candidates: list[str] = []
        for function_definition in available_functions:
            parameters: dict[str, object] = {}
            for name, definition in function_definition.parameters.items():
                parameters[name] = self._default_parameter_value(
                    definition.type
                )

            candidates.append(
                json.dumps(
                    {
                        "name": function_definition.name,
                        "parameters": parameters,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        return candidates

    def extract_string_candidates(self, prompt: str) -> list[str]:
        candidates: list[str] = []

        quoted_matches = re.findall(r'"([^"\\]+)"|\'([^\'\\]+)\'', prompt)
        for left, right in quoted_matches:
            value = left if left else right
            cleaned = value.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        for token in re.findall(r"[A-Za-z][A-Za-z\-']*", prompt):
            if token not in candidates:
                candidates.append(token)

        if not candidates:
            candidates.append("")
        return candidates

    def extract_number_candidates(self, prompt: str) -> list[float]:
        candidates: list[float] = []
        for match in re.findall(r"-?\d+(?:\.\d+)?", prompt):
            value = float(match)
            if value not in candidates:
                candidates.append(value)

        if not candidates:
            candidates.append(0.0)
        return candidates

    def parameter_candidates(
        self,
        prompt: str,
        parameter_definition: ParameterDefinition,
    ) -> list[object]:
        if parameter_definition.type == "string":
            return list(self.extract_string_candidates(prompt))
        if parameter_definition.type == "number":
            return list(self.extract_number_candidates(prompt))
        return [self._default_parameter_value(parameter_definition.type)]

    def materialize_candidate_json(
        self,
        function_name: str,
        parameters: dict[str, object],
    ) -> str:
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
    ) -> list[str]:
        parameter_names = list(function_definition.parameters.keys())
        value_space: dict[str, list[object]] = {}
        for name in parameter_names:
            definition = function_definition.parameters[name]
            values = self.parameter_candidates(prompt, definition)
            if not values:
                values = [self._default_parameter_value(definition.type)]
            value_space[name] = values

        expanded: list[dict[str, object]] = [{}]
        for parameter_name in parameter_names:
            next_expanded: list[dict[str, object]] = []
            for partial in expanded:
                for value in value_space[parameter_name]:
                    merged = dict(partial)
                    merged[parameter_name] = value
                    next_expanded.append(merged)

            expanded = next_expanded[:max_candidates_per_function]

        if not expanded:
            fallback_parameters: dict[str, object] = {}
            for name in parameter_names:
                fallback_parameters[name] = self._default_parameter_value(
                    function_definition.parameters[name].type
                )
            expanded = [fallback_parameters]

        candidate_texts: list[str] = []
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
    ) -> list[str]:
        all_candidates: list[str] = []
        for function_definition in available_functions:
            all_candidates.extend(
                self.expand_function_candidates_for_prompt(
                    function_definition=function_definition,
                    prompt=prompt,
                    max_candidates_per_function=max_candidates_per_function,
                )
            )
        return all_candidates
