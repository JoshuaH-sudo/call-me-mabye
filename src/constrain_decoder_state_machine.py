import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from llm_sdk import Small_LLM_Model

from .constrain_decoder import FunctionDefinition


_NUMBER_CHARS = set("-+0123456789.eE")
_NUMBER_COMPLETE_RE = re.compile(
    r"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)
_NUMBER_PREFIX_RE = re.compile(
    r"^(?:"
    r""  # empty buffer before any number char
    r"|-"  # just minus sign
    r"|-?(?:0|[1-9]\d*)"  # integer part
    r"|-?(?:0|[1-9]\d*)\."  # trailing dot waiting for decimals
    r"|-?(?:0|[1-9]\d*)\.\d*"  # decimal part in progress
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?[eE]"  # exponent marker
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]"  # exponent sign
    r"|-?(?:0|[1-9]\d*)(?:\.\d+)?[eE][+-]?\d+"  # exponent digits
    r")$"
)


class StateMachineConstrainedDecoder(BaseModel):
    """
    Constrained decoder implemented as a JSON grammar state machine.

    This decoder does NOT precompute full output candidates with fixed values.
    Instead, it enforces JSON shape and parameter types token-by-token:
    - function name must be one of available definitions
    - parameter keys must match the selected function exactly
    - parameter values must match expected JSON type (string/number)

    Output shape is always compact JSON:
    {"name":"...","parameters":{...}}
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    available_functions: list[FunctionDefinition]
    llm: Small_LLM_Model
    max_new_tokens: int = 512

    _segments_by_function: dict[str, list[tuple[str, str]]] = PrivateAttr(
        default_factory=dict
    )
    _token_text_cache: dict[int, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self._segments_by_function = {}
        for function_definition in self.available_functions:
            self._segments_by_function[function_definition.name] = (
                self._build_segments(function_definition)
            )

    def _build_segments(
        self,
        function_definition: FunctionDefinition,
    ) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        segments.append(
            (
                "literal",
                '{"name":"'
                + function_definition.name
                + '","parameters":{',
            )
        )

        parameter_names = list(function_definition.parameters.keys())
        for index, parameter_name in enumerate(parameter_names):
            segments.append(("literal", f'"{parameter_name}":'))
            parameter_type = function_definition.parameters[
                parameter_name
            ].type
            if parameter_type == "string":
                segments.append(("string", ""))
            elif parameter_type == "number":
                segments.append(("number", ""))
            else:
                raise RuntimeError(
                    "unsupported parameter type for constrained decoding: "
                    f"{parameter_type}"
                )

            if index != len(parameter_names) - 1:
                segments.append(("literal", ","))

        segments.append(("literal", "}}"))
        return segments

    def _initial_states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for function_name in self._segments_by_function.keys():
            states.append(
                {
                    "function_name": function_name,
                    "segment_index": 0,
                    "literal_offset": 0,
                    "string_started": False,
                    "string_escaped": False,
                    "number_buffer": "",
                    "complete": False,
                }
            )
        return states

    def _state_key(self, state: dict[str, Any]) -> tuple[Any, ...]:
        return (
            state["function_name"],
            state["segment_index"],
            state["literal_offset"],
            state["string_started"],
            state["string_escaped"],
            state["number_buffer"],
            state["complete"],
        )

    def _token_text(self, token_id: int) -> str:
        cached = self._token_text_cache.get(token_id)
        if cached is not None:
            return cached

        token_text = self.llm.decode([token_id])
        self._token_text_cache[token_id] = token_text
        return token_text

    def _is_complete_number(self, text: str) -> bool:
        return bool(_NUMBER_COMPLETE_RE.fullmatch(text))

    def _is_number_prefix(self, text: str) -> bool:
        return bool(_NUMBER_PREFIX_RE.fullmatch(text))

    def _consume_chunk(
        self,
        state: dict[str, Any],
        chunk: str,
    ) -> dict[str, Any] | None:
        next_state = dict(state)
        if next_state["complete"]:
            return None

        segments = self._segments_by_function[next_state["function_name"]]
        index = 0

        while index < len(chunk):
            segment_index = next_state["segment_index"]
            if segment_index >= len(segments):
                return None

            segment_kind, segment_value = segments[segment_index]

            if segment_kind == "literal":
                literal_offset = next_state["literal_offset"]
                if literal_offset >= len(segment_value):
                    next_state["segment_index"] = segment_index + 1
                    next_state["literal_offset"] = 0
                    continue

                remaining_literal = segment_value[literal_offset:]
                if not remaining_literal.startswith(chunk[index]):
                    return None

                matched = 0
                while (
                    index + matched < len(chunk)
                    and literal_offset + matched < len(segment_value)
                    and chunk[index + matched]
                    == segment_value[literal_offset + matched]
                ):
                    matched += 1

                next_state["literal_offset"] = literal_offset + matched
                index += matched

                if next_state["literal_offset"] == len(segment_value):
                    next_state["segment_index"] = segment_index + 1
                    next_state["literal_offset"] = 0
                    continue

                continue

            if segment_kind == "string":
                if not next_state["string_started"]:
                    if chunk[index] != '"':
                        return None
                    next_state["string_started"] = True
                    index += 1
                    continue

                if next_state["string_escaped"]:
                    next_state["string_escaped"] = False
                    index += 1
                    continue

                character = chunk[index]
                if character == "\\":
                    next_state["string_escaped"] = True
                    index += 1
                    continue

                if character == '"':
                    next_state["segment_index"] = segment_index + 1
                    next_state["string_started"] = False
                    next_state["string_escaped"] = False
                    index += 1
                    continue

                if ord(character) < 0x20:
                    return None

                index += 1
                continue

            if segment_kind == "number":
                character = chunk[index]
                if character in _NUMBER_CHARS:
                    next_state["number_buffer"] += character
                    index += 1
                    continue

                if self._is_complete_number(next_state["number_buffer"]):
                    next_state["segment_index"] = segment_index + 1
                    next_state["number_buffer"] = ""
                    continue

                return None

            return None

        segment_index = next_state["segment_index"]
        if segment_index < len(segments):
            segment_kind, _ = segments[segment_index]
            if segment_kind == "number" and not self._is_number_prefix(
                next_state["number_buffer"]
            ):
                return None

        if next_state["segment_index"] == len(segments):
            next_state["complete"] = True

        return next_state

    def _allowed_token_transitions(
        self,
        states: list[dict[str, Any]],
        prefix_ids: list[int],
    ) -> tuple[list[int], dict[int, list[dict[str, Any]]], list[float]]:
        logits = self.llm.get_logits_from_input_ids(prefix_ids)
        transitions: dict[int, list[dict[str, Any]]] = {}

        for token_id in range(len(logits)):
            token_text = self._token_text(token_id)
            if not token_text:
                continue

            next_states_for_token: dict[tuple[Any, ...], dict[str, Any]] = {}
            for state in states:
                next_state = self._consume_chunk(state, token_text)
                if next_state is None:
                    continue
                next_states_for_token[self._state_key(next_state)] = next_state

            if next_states_for_token:
                transitions[token_id] = list(next_states_for_token.values())

        allowed_token_ids = list(transitions.keys())
        if not allowed_token_ids:
            raise RuntimeError(
                "no valid constrained JSON continuation available"
            )

        return allowed_token_ids, transitions, logits

    def _select_best_token(
        self,
        allowed_token_ids: list[int],
        logits: list[float],
    ) -> int:
        for allowed_token_id in allowed_token_ids:
            if allowed_token_id < 0 or allowed_token_id >= len(logits):
                raise RuntimeError(
                    "allowed token id is out of vocabulary bounds"
                )
        return max(allowed_token_ids, key=lambda token_id: logits[token_id])

    def _is_complete(self, states: list[dict[str, Any]]) -> bool:
        return any(state["complete"] for state in states)

    def force_json_output(self, prefix_input_ids: list[int]) -> str:
        generated_ids: list[int] = []
        rolling_prefix = list(prefix_input_ids)
        states = self._initial_states()

        for _ in range(self.max_new_tokens):
            if self._is_complete(states):
                break

            allowed_token_ids, transitions, logits = (
                self._allowed_token_transitions(states, rolling_prefix)
            )
            selected_token_id = self._select_best_token(
                allowed_token_ids,
                logits,
            )

            generated_ids.append(selected_token_id)
            rolling_prefix.append(selected_token_id)
            states = transitions[selected_token_id]

        if not self._is_complete(states):
            raise RuntimeError(
                "state-machine constrained decoding reached max_new_tokens "
                "without completing valid JSON output"
            )

        output_text = self.llm.decode(generated_ids)
        self._validate_output_text(output_text)
        return output_text

    def _validate_output_text(self, output_text: str) -> None:
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "decoder produced invalid JSON output"
            ) from error

        if not isinstance(payload, dict):
            raise RuntimeError("decoder output JSON must be an object")

        if set(payload.keys()) != {"name", "parameters"}:
            raise RuntimeError(
                "decoder output must contain exactly 'name' and 'parameters'"
            )

        name = payload.get("name")
        parameters = payload.get("parameters")
        if not isinstance(name, str):
            raise RuntimeError("decoder output 'name' must be a string")
        if not isinstance(parameters, dict):
            raise RuntimeError(
                "decoder output 'parameters' must be an object"
            )

        function_definition = next(
            (
                function
                for function in self.available_functions
                if function.name == name
            ),
            None,
        )
        if function_definition is None:
            raise RuntimeError("decoder selected unknown function name")

        expected_parameter_names = set(function_definition.parameters.keys())
        if set(parameters.keys()) != expected_parameter_names:
            raise RuntimeError(
                "decoder output parameters do not match function definition"
            )

        for parameter_name, parameter_definition in (
            function_definition.parameters.items()
        ):
            value = parameters[parameter_name]
            if parameter_definition.type == "string" and not isinstance(
                value, str
            ):
                raise RuntimeError(
                    "decoder output parameter type mismatch: expected string"
                )
            if parameter_definition.type == "number" and not isinstance(
                value,
                (int, float),
            ):
                raise RuntimeError(
                    "decoder output parameter type mismatch: expected number"
                )

    def decode(self, input_ids: Any) -> str:
        prefix_ids = input_ids[0].tolist()
        if not isinstance(prefix_ids, list):
            raise RuntimeError("input_ids must contain a list of token ids")
        return self.force_json_output(prefix_ids)
