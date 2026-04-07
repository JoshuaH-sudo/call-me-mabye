import json

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
    """

    available_functions: list[FunctionDefinition]
    encoded_function_definitions: list[Tensor]

    def __init__(
        self,
        available_functions: list[FunctionDefinition],
        llm: Small_LLM_Model,
    ):
        self.available_functions = available_functions
        self.encoded_function_definitions = [
            llm.encode(json.dumps(item.model_dump(), sort_keys=True))
            for item in available_functions
        ]

    def decode(self, input_ids: Tensor) -> str:
        # Placeholder for constrained decoding logic.
        return (
            "Decoded output based on input_ids with "
            f"{len(self.encoded_function_definitions)} constraints"
        )
