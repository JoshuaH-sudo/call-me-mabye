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
    encoded_function_definitions: list[Tensor]
    llm: Small_LLM_Model

    def __init__(
        self,
        available_functions: list[FunctionDefinition],
        llm: Small_LLM_Model,
    ):
        self.llm = llm
        self.available_functions = available_functions
        self.encoded_function_definitions = [
            self.llm.encode(json.dumps(item.model_dump(), sort_keys=True))
            for item in available_functions
        ]

    def _force_token(
        self,
        prefix_ids: list[int],
        allowed_token_id: int,
    ) -> int:
        # Get next-token logits from the model for the current prefix.
        logits = self.llm.get_logits_from_input_ids(prefix_ids)
        if allowed_token_id < 0 or allowed_token_id >= len(logits):
            raise RuntimeError("allowed token id is out of vocabulary bounds")

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
        constrained_logits[allowed_token_id] = logits[allowed_token_id]

        # Greedy selection on constrained logits always returns
        # allowed_token_id.
        return max(
            range(len(constrained_logits)),
            key=lambda index: constrained_logits[index],
        )

    def generate_wrapped_text(
        self,
        prefix_input_ids: list[int],
        original_text: str,
    ) -> str:
        """
        Basic constrained-decoding demo that forces output to:
        * {original_text} *
        """
        target_text = f"* {original_text} *"
        encoded_target = self.llm.encode(target_text)
        target_ids = cast(list[int], encoded_target[0].tolist())

        # How text becomes constrained generation target:
        #
        # target_text      = "* original given text *"
        # target token ids = [t0, t1, t2, ..., tn]
        #
        # We then force generation of t0, then t1, ..., then tn.
        # After each forced token, we append it to rolling_prefix so the next
        # logits are conditioned on all previously generated tokens.

        generated_ids: list[int] = []
        rolling_prefix = list(prefix_input_ids)
        for required_token_id in target_ids:
            selected_token_id = self._force_token(
                prefix_ids=rolling_prefix,
                allowed_token_id=required_token_id,
            )
            # Keep track of generated output token ids.
            generated_ids.append(selected_token_id)
            # Update model context for the next generation step.
            rolling_prefix.append(selected_token_id)

        # Decode only generated ids so output is exactly constrained segment.
        return self.llm.decode(generated_ids)

    def decode(self, input_ids: Tensor) -> str:
        # Keep decode as a simple wrapper for the basic demonstration.
        prefix_ids = cast(list[int], input_ids[0].tolist())
        wrapped_text = self.generate_wrapped_text(
            prefix_input_ids=prefix_ids,
            original_text="original given text",
        )
        return (
            "Decoded output based on input_ids with constrained format: "
            f"{wrapped_text}"
        )
