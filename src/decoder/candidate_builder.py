"""JSON candidate builder for constrained decoding.

The :class:`CandidateBuilder` is responsible for two things:

1. **Candidate enumeration** — for every available function, extract
   plausible parameter values from the prompt and serialize every valid
   combination as a compact JSON string that the decoder can treat as a
   decoding target.
2. **LLM-driven function selection** — by including candidates from *all*
   available functions, the constrained decoder's token-level logit scoring
   (not any heuristic) determines which function is chosen.

Design notes
------------
* The candidate set is *finite and precomputed*, which is what makes
  constrained decoding possible: we encode each candidate once and then do
  cheap prefix-match comparisons at every generation step.
* Cross-product expansion is capped at ``max_candidates_per_function`` to
  keep memory and encoding time bounded.
* When all parameters are numeric, a *sliding-window* strategy is used
  instead of a cross-product to preserve the natural left-to-right ordering
  of numbers mentioned in the prompt.
"""
import json

from .models import FunctionDefinition, ParameterDefinition
from .extractors.number import NumberParameterExtractor
from .extractors.regex import RegexParameterExtractor
from .extractors.string import StringParameterExtractor
from .types import (
    OutputCandidate,
    OutputCandidates,
    ParameterValue,
    ParameterValues,
    ParameterValueSpace,
)


class CandidateBuilder:
    """Builds compact JSON function-call candidates from function schemas.

    Coordinates parameter extraction and JSON candidate generation for all
    available functions.  By generating candidates for every function, the
    constrained decoder's LLM-driven logit scoring — not any heuristic —
    determines which function is ultimately selected.

    Delegates extraction logic to three specialised extractor classes:

    * :class:`~src.decoder.extractors.string.StringParameterExtractor`
    * :class:`~src.decoder.extractors.number.NumberParameterExtractor`
    * :class:`~src.decoder.extractors.regex.RegexParameterExtractor`
    """

    def __init__(self) -> None:
        """Instantiate all three parameter extractors."""
        self.string_extractor = StringParameterExtractor()
        self.number_extractor = NumberParameterExtractor()
        self.regex_extractor = RegexParameterExtractor()

    def _default_parameter_value(self, parameter_type: str) -> object:
        """Return the safe fallback value for *parameter_type*.

        Used when no candidates can be extracted from the prompt so that
        the candidate list is never empty and decoding can always proceed.

        Args:
            parameter_type: Schema type string (``"string"`` or ``"number"``).

        Returns:
            ``""`` for string parameters, ``0`` for number parameters.

        Raises:
            RuntimeError: If *parameter_type* is not a supported type.
        """
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
        """Extract candidate values for a single parameter from *prompt*.

        Dispatch rules (checked in order):

        1. If *parameter_name* is ``"regex"``, use
           :class:`~src.decoder.extractors.regex.RegexParameterExtractor`
           regardless of the declared type.
        2. If *parameter_definition.type* is ``"string"``, use
           :class:`~src.decoder.extractors.string.StringParameterExtractor`.
        3. If *parameter_definition.type* is ``"number"``, use
           :class:`~src.decoder.extractors.number.NumberParameterExtractor`.
        4. Otherwise fall back to the default value for that type.

        Args:
            prompt: The user prompt to extract values from.
            parameter_definition: Schema for the parameter being extracted.
            parameter_name: The parameter's name in the function schema
                (used for name-based dispatch and context-aware extraction).

        Returns:
            A list of candidate values ordered by extraction priority.
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
        """Serialize a function name + parameter map to a compact JSON string.

        Keys are sorted alphabetically to ensure a deterministic byte
        sequence regardless of the insertion order of *parameters*.

        Args:
            function_name: The function name to embed as ``"name"``.
            parameters: The parameter name → value mapping.

        Returns:
            A compact JSON string, e.g.
            ``'{"name":"add","parameters":{"a":1,"b":2}}'``.
        """
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
        """Build all JSON candidate strings for one function and prompt.

        Two strategies are used depending on the parameter types:

        **Sliding-window (all-numeric parameters)**
        When every parameter expects a number, the extracted values are
        assigned left-to-right using a sliding window over the ordered list
        of mentions.  This mirrors the natural ordering of numbers in prose
        (e.g. "add 3 and 7" → ``a=3, b=7``) and avoids the cross-product
        explosion for multi-parameter numeric functions.

        **Cross-product (mixed / string parameters)**
        For functions with string (or mixed) parameters the method builds
        one candidate per combination of extracted values across all
        parameters.  The expansion is capped at *max_candidates_per_function*
        to bound the encoding cost.

        Args:
            function_definition: The function to build candidates for.
            prompt: The user prompt used for parameter extraction.
            max_candidates_per_function: Maximum number of candidate strings
                to return (duplicates are removed before the cap is applied).

        Returns:
            A deduplicated list of compact JSON candidate strings.
        """
        parameter_names = list(function_definition.parameters.keys())

        # ------------------------------------------------------------------ #
        # Strategy A: sliding-window for all-numeric parameter lists          #
        # ------------------------------------------------------------------ #
        if parameter_names and all(
            function_definition.parameters[name].type == "number"
            for name in parameter_names
        ):
            numeric_values = self.number_extractor.extract_candidates(prompt)
            if numeric_values:
                aligned: list[ParameterValues] = []
                width = len(parameter_names)

                if len(numeric_values) >= width:
                    # Slide a window of exactly `width` values across the
                    # extracted list, producing one candidate per position.
                    max_windows = len(numeric_values) - width + 1
                    for window_start in range(max_windows):
                        params: ParameterValues = {}
                        for offset, parameter_name in enumerate(
                            parameter_names
                        ):
                            params[parameter_name] = numeric_values[
                                window_start + offset
                            ]
                        aligned.append(params)
                else:
                    # Fewer values than parameters: fill what we have and pad
                    # the remainder with the default value (0).
                    params = {}
                    for offset, parameter_name in enumerate(parameter_names):
                        if offset < len(numeric_values):
                            params[parameter_name] = numeric_values[offset]
                        else:
                            params[parameter_name] = (
                                self._default_parameter_value("number")
                            )
                    aligned.append(params)

                # Deduplicate and serialize the aligned parameter maps.
                aligned_candidate_texts: OutputCandidates = []
                aligned_seen: set[str] = set()
                for parameters in aligned[:max_candidates_per_function]:
                    candidate = self.materialize_candidate_json(
                        function_name=function_definition.name,
                        parameters=parameters,
                    )
                    if candidate in aligned_seen:
                        continue
                    aligned_seen.add(candidate)
                    aligned_candidate_texts.append(candidate)

                if aligned_candidate_texts:
                    return aligned_candidate_texts

        # ------------------------------------------------------------------ #
        # Strategy B: cross-product expansion for mixed / string parameters   #
        # ------------------------------------------------------------------ #

        # Collect candidate value lists for each parameter.
        value_space: ParameterValueSpace = {}
        for name in parameter_names:
            definition = function_definition.parameters[name]
            values = self.parameter_candidates(prompt, definition, name)
            if not values:
                # Guarantee at least one value so the cross-product is
                # never empty.
                values = [self._default_parameter_value(definition.type)]
            value_space[name] = values

        # Iteratively extend the list of partial parameter dicts by adding
        # one more parameter at a time.  The cap is applied after each
        # extension to prevent combinatorial explosion.
        expanded: list[ParameterValues] = [{}]
        for parameter_name in parameter_names:
            next_expanded: list[ParameterValues] = []
            for partial in expanded:
                for value in value_space[parameter_name]:
                    merged = dict(partial)
                    merged[parameter_name] = value
                    next_expanded.append(merged)

            expanded = next_expanded[:max_candidates_per_function]

        # Safety net: if the expansion somehow ended up empty, emit one
        # candidate using all-default values.
        if not expanded:
            fallback_parameters: ParameterValues = {}
            for name in parameter_names:
                fallback_parameters[name] = self._default_parameter_value(
                    function_definition.parameters[name].type
                )
            expanded = [fallback_parameters]

        # Serialize and deduplicate.
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
        """Return JSON candidates for *every* available function.

        Candidates are generated for all functions so that the constrained
        decoder's token-level logit scoring — not any heuristic — determines
        which function is selected.  The model naturally assigns higher
        probability to the token sequence that best matches the prompt, and
        the prefix-matcher ensures only valid continuations are ever chosen.

        Args:
            available_functions: All function definitions to consider.
            prompt: The raw user prompt used for parameter extraction.
            max_candidates_per_function: Maximum number of candidate strings
                per function; passed through to the expansion step.

        Returns:
            A flat list of compact JSON candidate strings covering all
            available functions.
        """
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
