"""JSON candidate builder for constrained decoding.

The :class:`CandidateBuilder` is responsible for two things:

1. **Function selection** — given a prompt and all available function
   definitions, score each function by token overlap with the prompt and
   pick the best-matching one.
2. **Candidate enumeration** — for the selected function, extract plausible
   parameter values from the prompt and serialize every valid combination as
   a compact JSON string that the decoder can treat as a decoding target.

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
* ``regex``-typed parameters are always extracted by
  :class:`~src.decoder.regex_decoder.RegexConstrainedDecoder` when an LLM
  is provided (deferred import to avoid loading torch in schema-only
  contexts).  When no LLM is available the safe match-all ``".*"`` is used
  as a placeholder.
"""

import json
import re
from typing import TYPE_CHECKING

from .models import FunctionDefinition, ParameterDefinition
from .extractors.number import NumberParameterExtractor
from .extractors.string import StringParameterExtractor
from .types import (
    OutputCandidate,
    OutputCandidates,
    ParameterValue,
    ParameterValues,
    ParameterValueSpace,
)

if TYPE_CHECKING:
    # Imported here only to satisfy type checkers; the actual import is
    # deferred to __init__ to avoid loading the LLM stack when
    # CandidateBuilder is used without a model.
    from llm_sdk import Small_LLM_Model

# Words that carry no semantic signal for function selection.
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
    """Lowercase *text*, split on word boundaries, and remove stopwords.

    Returns a set so that callers can do fast intersection tests.
    """
    return {
        w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOPWORDS
    }


def _score_function(fn: FunctionDefinition, prompt_tokens: set[str]) -> int:
    """Return a relevance score for *fn* given the tokenised *prompt_tokens*.

    The score is the number of tokens shared between the prompt and the
    union of the function's name tokens and description tokens.  A small
    bonus is added for substitution-related functions when the prompt
    contains substitution intent words, improving selection accuracy for
    that common case.

    Args:
        fn: The function definition to score.
        prompt_tokens: Pre-tokenised (and stopword-filtered) prompt words.

    Returns:
        An integer relevance score; higher means a better match.
    """
    name_tokens = {
        p for p in fn.name.lower().split("_") if p not in _STOPWORDS
    }
    fn_tokens = _tokenize(fn.description) | name_tokens
    score = len(fn_tokens & prompt_tokens)

    # Boost substitution functions when the prompt uses "replace" or
    # "substitute" — these are strong signals that the user wants a
    # string-substitution function rather than a generic string function.
    if {"replace", "substitute"} & prompt_tokens and "substitute" in fn.name:
        score += 5
    return score


class CandidateBuilder:
    """Builds compact JSON function-call candidates from function schemas.

    Coordinates parameter extraction and JSON candidate generation.
    Delegates extraction logic to two specialised extractor classes:

    * :class:`~src.decoder.extractors.string.StringParameterExtractor`
    * :class:`~src.decoder.extractors.number.NumberParameterExtractor`

    ``regex``-named parameters are always handled by
    :class:`~src.decoder.regex_decoder.RegexConstrainedDecoder` when an
    LLM is provided at construction time.  When no LLM is available the
    safe match-all ``".*"`` is used as a placeholder.
    """

    def __init__(
        self,
        llm: "Small_LLM_Model | None" = None,
    ) -> None:
        """Instantiate the parameter extractors.

        Args:
            llm: Optional language model.  When provided, a
                :class:`~src.decoder.regex_decoder.RegexConstrainedDecoder`
                is created and used for ``regex``-named parameters.
                When ``None``, ``".*"`` is returned as a safe placeholder.
        """
        self.string_extractor = StringParameterExtractor()
        self.number_extractor = NumberParameterExtractor()
        self._regex_decoder: "RegexConstrainedDecoder | None" = None
        if llm is not None:
            # Deferred import to avoid loading torch/llm_sdk when this
            # class is used without a model (e.g. schema-only tooling).
            from .regex_decoder import RegexConstrainedDecoder
            self._regex_decoder = RegexConstrainedDecoder(llm=llm)

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
        function_name: str = "",
    ) -> list[ParameterValue]:
        """Extract candidate values for a single parameter from *prompt*.

        Dispatch rules (checked in order):

        1. If *parameter_name* is ``"regex"``, use
           :class:`~src.decoder.regex_decoder.RegexConstrainedDecoder`
           when an LLM is available; otherwise return ``[".*"]``.
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
            function_name: The name of the function being called.  Forwarded
                to the string extractor so that function-name tokens can be
                excluded from bare-word string candidates.

        Returns:
            A list of candidate values ordered by extraction priority.
        """
        if parameter_name == "regex":
            if self._regex_decoder is not None:
                return [self._regex_decoder.generate_regex(prompt)]
            # No LLM available — return a safe match-all placeholder.
            return [".*"]
        if parameter_definition.type == "string":
            return list(
                self.string_extractor.extract_candidates(
                    prompt,
                    parameter_name,
                    function_name,
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
        of mentions.  To avoid numeric context contamination from preamble
        sentences, extraction is attempted on the question/intent segment of
        the prompt first.  If that segment alone yields enough values to fill
        all parameters the window is built exclusively from those values,
        keeping the candidate set small and unambiguous.  Full-prompt
        extraction is used only as a fallback when the question segment does
        not contain enough numbers by itself.

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
            width = len(parameter_names)

            # Prefer numbers from the question/intent segment to avoid
            # contamination from numeric context in preamble sentences.
            question_values = (
                self.number_extractor.extract_candidates_from_question_segment(
                    prompt
                )
            )
            if len(question_values) >= width:
                numeric_values = question_values
            else:
                # Fall back to full-prompt extraction when the question
                # segment alone doesn't supply enough numbers.
                numeric_values = self.number_extractor.extract_candidates(
                    prompt
                )

            if numeric_values:
                aligned: list[ParameterValues] = []

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

        # Collect candidate value lists for each parameter, passing the
        # active function name so that the string extractor can exclude
        # function-name-derived tokens from its bare-word candidates.
        value_space: ParameterValueSpace = {}
        for name in parameter_names:
            definition = function_definition.parameters[name]
            values = self.parameter_candidates(
                prompt,
                definition,
                name,
                function_definition.name,
            )
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
        """Select the best function and return JSON candidates for *prompt*.

        Selection pipeline
        ------------------
        1. Tokenise the prompt (stopwords removed).
        2. Score every function via :func:`_score_function`.
        3. Filter out functions for which no parameter candidates can be
           extracted — they cannot produce usable JSON.
        4. Fall back to the top-scoring function if all are filtered out.
        5. Take only the single best-matching function and expand its
           candidates via :meth:`expand_function_candidates_for_prompt`.

        Args:
            available_functions: All function definitions to consider.
            prompt: The raw user prompt.
            max_candidates_per_function: Passed through to the expansion step.

        Returns:
            A list of compact JSON candidate strings for the selected function.
        """
        prompt_tokens = _tokenize(prompt)

        def has_candidates_for_all_params(fn: FunctionDefinition) -> bool:
            """Return True when every parameter yields at least one value."""
            return all(
                bool(self.parameter_candidates(prompt, defn, name, fn.name))
                for name, defn in fn.parameters.items()
            )

        # Rank all functions by relevance to the prompt.
        sorted_fns = sorted(
            available_functions,
            key=lambda fn: _score_function(fn, prompt_tokens),
            reverse=True,
        )

        # Keep only functions whose parameters can all be filled from the
        # prompt; this avoids emitting candidates with blank/default values
        # when a better-matched function exists.
        filtered_fns = [
            fn for fn in sorted_fns if has_candidates_for_all_params(fn)
        ]

        # If no function survives the filter, fall back to the top-ranked
        # function (even if some parameters will use defaults) to ensure we
        # always produce at least one candidate.
        if not filtered_fns:
            filtered_fns = sorted_fns[:1]
        else:
            # limit to top 3 functions to control expansion cost
            filtered_fns = filtered_fns

        all_candidates: OutputCandidates = []
        for function_definition in filtered_fns:
            all_candidates.extend(
                self.expand_function_candidates_for_prompt(
                    function_definition=function_definition,
                    prompt=prompt,
                    max_candidates_per_function=max_candidates_per_function,
                )
            )
        return all_candidates
