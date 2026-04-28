"""
Constrained JSON decoder for function-call output generation.

Uses a grammar state-machine to enforce both JSON structure and
parameter-type validity at token-selection time without pre-enumerating
all possible parameter values.

Output shape (always compact JSON):
    {"name":"<fn>","parameters":{"<p1>":<v1>,...}}

How it works
------------
Each function definition is compiled into a *segment sequence*: an
ordered list of alternating literal segments (fixed text that must appear
verbatim) and parameter segments (typed values decoded by a
ParameterSegmentHandler).

At every generation step the decoder maintains a list of live *state
threads*, one per function candidate.  For every vocabulary token it
tries to advance each thread.  Threads that reject the token are pruned.
Model logits then pick the best surviving token, and the process repeats
until one thread reaches a complete state.

Functions whose parameters include unsupported types are silently skipped
during initialisation (see ``skipped_functions``).  New types are
supported by adding a ParameterSegmentHandler subclass and registering it
in create_parameter_handler().
"""
import json
from typing import Any, cast

from llm_sdk import Small_LLM_Model

from .decoder_pipeline import (
    FunctionDefinition,
    ParameterDefinition,
    ReturnDefinition,
)
from .decoder_pipeline.parameter_handlers import (
    ParameterSegmentHandler,
    UnsupportedParameterHandler,
    create_parameter_handler,
)

__all__ = [
    "ParameterDefinition",
    "ReturnDefinition",
    "FunctionDefinition",
    "ConstrainedDecoder",
]

# A segment is either ("literal", fixed_text) or ("param", handler).
_Segment = tuple[str, str | ParameterSegmentHandler]


class ConstrainedDecoder:
    """
    Decodes model output to a valid JSON function-call object.

    Parameters
    ----------
    available_functions:
        All known function definitions.  Functions with unsupported
        parameter types are automatically excluded from decoding; their
        names are recorded in ``skipped_functions``.
    llm:
        The language model used for logit scoring.
    max_new_tokens:
        Hard upper bound on generated tokens before raising an error.
    """

    def __init__(
        self,
        available_functions: list[FunctionDefinition],
        llm: Small_LLM_Model,
        max_new_tokens: int = 512,
    ) -> None:
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self._token_text_cache: dict[int, str] = {}

        # Compile segment sequences; skip unsupported functions.
        self._segments_by_function: dict[str, list[_Segment]] = {}
        self.skipped_functions: list[str] = []
        for fn in available_functions:
            segments = self._build_segments(fn)
            if segments is not None:
                self._segments_by_function[fn.name] = segments
            else:
                self.skipped_functions.append(fn.name)

    # ------------------------------------------------------------------
    # Segment compilation
    # ------------------------------------------------------------------

    def _build_segments(
        self,
        fn: FunctionDefinition,
    ) -> list[_Segment] | None:
        """
        Compile a function definition into a segment sequence.

        Returns None when any parameter uses an unsupported type, which
        causes the function to be excluded from decoding.
        """
        segments: list[_Segment] = []
        # Opening literal common to all output for this function.
        opening = '{"name":"' + fn.name + '","parameters":{'
        segments.append(("literal", opening))

        param_names = list(fn.parameters.keys())
        for i, param_name in enumerate(param_names):
            segments.append(("literal", f'"{param_name}":'))

            handler = create_parameter_handler(
                fn.parameters[param_name].type
            )
            if isinstance(handler, UnsupportedParameterHandler):
                # Skip the whole function rather than decoding partially.
                return None
            segments.append(("param", handler))

            if i != len(param_names) - 1:
                segments.append(("literal", ","))

        segments.append(("literal", "}}"))
        return segments

    # ------------------------------------------------------------------
    # Token text cache
    # ------------------------------------------------------------------

    def _token_text(self, token_id: int) -> str:
        cached = self._token_text_cache.get(token_id)
        if cached is not None:
            return cached
        text = cast(str, self.llm.decode([token_id]))
        self._token_text_cache[token_id] = text
        return text

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _initial_states(self) -> list[dict[str, Any]]:
        """One starting state thread per available function."""
        return [
            {
                "function_name": fn_name,
                "segment_index": 0,
                "literal_offset": 0,
                # None when not currently inside a param segment.
                "param_state": None,
                "complete": False,
            }
            for fn_name in self._segments_by_function
        ]

    def _state_key(self, state: dict[str, Any]) -> tuple[Any, ...]:
        """Hashable key used to deduplicate equivalent state threads."""
        param_state = state["param_state"]
        param_key = (
            json.dumps(param_state, sort_keys=True)
            if param_state is not None
            else ""
        )
        return (
            state["function_name"],
            state["segment_index"],
            state["literal_offset"],
            param_key,
            state["complete"],
        )

    def _consume_chunk(
        self,
        state: dict[str, Any],
        chunk: str,
    ) -> dict[str, Any] | None:
        """
        Try to advance a state thread by consuming the token text.

        Returns the updated state if the chunk is a valid continuation,
        or None if it is not.

        The algorithm walks through the segment sequence for the thread's
        function, consuming characters from *chunk* one segment at a time:

        - Literal segments: characters must match the fixed text exactly.
        - Param segments: characters are delegated to the handler.  When
          the handler signals completion the decoder advances to the next
          segment and continues processing any remaining characters.
        """
        if state["complete"]:
            return None

        # Copy the top-level state dict. param_state is a dict whose
        # values are immutable scalars (bool, str, int), so a shallow copy
        # of param_state is sufficient to prevent mutation of the original.
        s: dict[str, Any] = {
            "function_name": state["function_name"],
            "segment_index": state["segment_index"],
            "literal_offset": state["literal_offset"],
            "param_state": (
                dict(state["param_state"])
                if state["param_state"] is not None
                else None
            ),
            "complete": state["complete"],
        }

        segments = self._segments_by_function[s["function_name"]]
        index = 0

        while index < len(chunk):
            seg_idx = s["segment_index"]
            if seg_idx >= len(segments):
                return None

            kind, seg_value = segments[seg_idx]

            if kind == "literal":
                lit = cast(str, seg_value)
                offset = s["literal_offset"]

                if offset >= len(lit):
                    # Literal exhausted; advance to the next segment.
                    s["segment_index"] = seg_idx + 1
                    s["literal_offset"] = 0
                    s["param_state"] = None
                    continue

                if chunk[index] != lit[offset]:
                    return None

                # Consume as many matching literal characters as possible
                # in one pass to minimise loop overhead.
                matched = 0
                while (
                    index + matched < len(chunk)
                    and offset + matched < len(lit)
                    and chunk[index + matched] == lit[offset + matched]
                ):
                    matched += 1

                s["literal_offset"] = offset + matched
                index += matched

                if s["literal_offset"] == len(lit):
                    s["segment_index"] = seg_idx + 1
                    s["literal_offset"] = 0
                    s["param_state"] = None
                continue

            if kind == "param":
                handler = cast(ParameterSegmentHandler, seg_value)

                # Initialise handler state on first entry into this seg.
                if s["param_state"] is None:
                    s["param_state"] = handler.initial_state()

                # Defensive: if already complete here, advance segment.
                if handler.is_complete(s["param_state"]):
                    s["segment_index"] = seg_idx + 1
                    s["literal_offset"] = 0
                    s["param_state"] = None
                    continue

                updated, new_index = handler.consume_chunk(
                    s["param_state"], chunk, index
                )
                if updated is None:
                    return None

                s["param_state"] = updated

                if handler.is_complete(updated):
                    # Value fully decoded; advance to next segment.
                    # new_index is the first char NOT consumed by the
                    # handler (e.g. the "," or "}" following a number).
                    s["segment_index"] = seg_idx + 1
                    s["literal_offset"] = 0
                    s["param_state"] = None
                    index = new_index
                    continue

                # Handler still accumulating; must have reached chunk end.
                if new_index < len(chunk):
                    return None
                index = new_index
                continue

            return None  # Unknown segment kind (defensive)

        # After consuming the full chunk, validate any in-progress param
        # segment.  Numbers need an explicit prefix check because they
        # lack a terminal character; a buffer like "-" cannot lead to a
        # valid value.
        seg_idx = s["segment_index"]
        if seg_idx < len(segments):
            kind, seg_value = segments[seg_idx]
            if kind == "param" and s["param_state"] is not None:
                handler = cast(ParameterSegmentHandler, seg_value)
                if not handler.is_valid_prefix(s["param_state"]):
                    return None

        if s["segment_index"] == len(segments):
            s["complete"] = True

        return s

    def _allowed_token_transitions(
        self,
        states: list[dict[str, Any]],
        prefix_ids: list[int],
    ) -> tuple[list[int], dict[int, list[dict[str, Any]]], list[float]]:
        """
        Score every vocabulary token and collect those that advance at
        least one live state thread.

        Returns:
            allowed      — token ids with at least one valid next state.
            transitions  — token id -> list of resulting next states.
            logits       — raw model logits for the current prefix.
        """
        raw = self.llm.get_logits_from_input_ids(prefix_ids)
        logits = cast(list[float], raw)
        transitions: dict[int, list[dict[str, Any]]] = {}

        for token_id in range(len(logits)):
            token_text = self._token_text(token_id)
            if not token_text:
                continue

            # Try the token against every live state; deduplicate results.
            next_states: dict[tuple[Any, ...], dict[str, Any]] = {}
            for state in states:
                next_state = self._consume_chunk(state, token_text)
                if next_state is None:
                    continue
                next_states[self._state_key(next_state)] = next_state

            if next_states:
                transitions[token_id] = list(next_states.values())

        allowed = list(transitions.keys())
        if not allowed:
            raise RuntimeError(
                "no valid constrained JSON continuation available"
            )
        return allowed, transitions, logits

    def _is_complete(self, states: list[dict[str, Any]]) -> bool:
        return any(state["complete"] for state in states)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def force_json_output(self, prefix_input_ids: list[int]) -> str:
        """
        Generate a complete JSON function-call string constrained to the
        available function definitions.

        Generation loop:
        1. Get model logits for the current rolling prefix.
        2. Collect every vocabulary token that advances at least one live
           state thread.
        3. Pick the token with the highest logit among those.
        4. Append it and repeat.
        5. Stop when any thread reaches a complete state.
        """
        if not self._segments_by_function:
            raise RuntimeError(
                "no supported functions available for constrained decoding"
            )

        generated_ids: list[int] = []
        rolling_prefix = list(prefix_input_ids)
        states = self._initial_states()

        for _ in range(self.max_new_tokens):
            if self._is_complete(states):
                break

            allowed, transitions, logits = (
                self._allowed_token_transitions(states, rolling_prefix)
            )
            selected = max(allowed, key=lambda tid: logits[tid])

            generated_ids.append(selected)
            rolling_prefix.append(selected)
            states = transitions[selected]

        if not self._is_complete(states):
            raise RuntimeError(
                "constrained decoding reached max_new_tokens without "
                "completing valid JSON output"
            )

        return cast(str, self.llm.decode(generated_ids))

    def decode(self, input_ids: Any) -> str:
        """Convenience wrapper that unpacks a batched input tensor."""
        prefix_ids = cast(list[int], input_ids[0].tolist())
        return self.force_json_output(prefix_ids)
