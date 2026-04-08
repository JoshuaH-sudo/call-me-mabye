import json
from typing import cast

from torch import Tensor
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class ReturnDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class ConstrainedDecoder:
    """
    Decodes model outputs while constraining choices
    to known function definitions.

    Token and ID flow (high level):

    text -> tokenizer -> token_ids -> model logits -> constrained choice

    Example:
    "* hello *" -> [42, 991, 17] -> logits over vocab -> keep only one id
    """

    available_functions: list[FunctionDefinition]
    encoded_output_candidates: list[list[int]]
    llm: Small_LLM_Model

    def __init__(
        self,
        available_functions: list[FunctionDefinition],
        llm: Small_LLM_Model,
    ):
        self.llm = llm
        self.available_functions = available_functions
        self.encoded_output_candidates = [
            self._encode_text(candidate)
            for candidate in self._build_output_candidates(available_functions)
        ]

    def _encode_text(self, text: str) -> list[int]:
        return cast(list[int], self.llm.encode(text)[0].tolist())

    def _default_parameter_value(self, parameter_type: str) -> object:
        if parameter_type == "string":
            return ""
        if parameter_type == "number":
            return 0
        raise RuntimeError(
            "unsupported parameter type for constrained decoding: "
            f"{parameter_type}"
        )

    def _build_output_candidates(
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

    def _next_allowed_token_ids(
        self,
        generated_ids: list[int],
    ) -> list[int]:
        allowed_token_ids: list[int] = []
        seen_token_ids: set[int] = set()
        for candidate_ids in self.encoded_output_candidates:
            prefix_length = len(generated_ids)
            if prefix_length >= len(candidate_ids):
                continue
            if candidate_ids[:prefix_length] != generated_ids:
                continue

            next_token_id = candidate_ids[prefix_length]
            if next_token_id in seen_token_ids:
                continue

            seen_token_ids.add(next_token_id)
            allowed_token_ids.append(next_token_id)

        if not allowed_token_ids:
            raise RuntimeError(
                "no valid constrained JSON continuation available"
            )
        return allowed_token_ids

    def _is_complete_output(self, generated_ids: list[int]) -> bool:
        return any(
            generated_ids == candidate_ids
            for candidate_ids in self.encoded_output_candidates
        )

    def _force_token(
        self,
        prefix_ids: list[int],
        allowed_token_ids: list[int],
    ) -> int:
        # Get next-token logits from the model for the current prefix.
        logits = self.llm.get_logits_from_input_ids(prefix_ids)
        for allowed_token_id in allowed_token_ids:
            if allowed_token_id < 0 or allowed_token_id >= len(logits):
                raise RuntimeError(
                    "allowed token id is out of vocabulary bounds"
                )

        # Constrained decoding at one generation step:
        #
        # vocab ids:      [0, 1, 2, 3, 4, ...]
        # raw logits:     [a, b, c, d, e, ...]
        # allowed id:                 3
        # masked logits:  [-inf, -inf, -inf, d, -inf, ...]
        # selected id:                3
        #
        # This enforces a single valid next token.
        constrained_logits = [float("-inf")] * len(logits)
        for allowed_token_id in allowed_token_ids:
            constrained_logits[allowed_token_id] = logits[allowed_token_id]

        # Greedy selection on constrained logits always returns
        # allowed_token_id.
        return max(
            range(len(constrained_logits)),
            key=lambda index: constrained_logits[index],
        )

    def force_json_output(
        self,
        prefix_input_ids: list[int],
    ) -> str:
        generated_ids: list[int] = []
        rolling_prefix = list(prefix_input_ids)

        while not self._is_complete_output(generated_ids):
            allowed_token_ids = self._next_allowed_token_ids(generated_ids)
            selected_token_id = self._force_token(
                prefix_ids=rolling_prefix,
                allowed_token_ids=allowed_token_ids,
            )
            generated_ids.append(selected_token_id)
            rolling_prefix.append(selected_token_id)

        return self.llm.decode(generated_ids)

    def decode(self, input_ids: Tensor) -> str:
        prefix_ids = cast(list[int], input_ids[0].tolist())
        return self.force_json_output(prefix_ids)
