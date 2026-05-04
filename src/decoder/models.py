"""Pydantic models that describe the function-call schema loaded from disk.

These models are the *decoder-internal* representation of what functions are
available.  They are distinct from the output models in ``src.models`` so
that changes to the public output format do not inadvertently affect how
schemas are parsed or how candidate values are generated.
"""
from pydantic import BaseModel, ConfigDict


class ParameterDefinition(BaseModel):
    """Schema for a single function parameter.

    Attributes:
        type: The parameter's type string as declared in the definitions file.
            Currently supported values are ``"string"``, ``"number"``, and
            ``"integer"``.
    """

    model_config = ConfigDict(extra="forbid")

    type: str


class ReturnDefinition(BaseModel):
    """Schema for a function's return value.

    Attributes:
        type: The return type string (e.g. ``"string"``, ``"number"``,
            ``"integer"``).
            Not used during decoding but preserved for completeness.
    """

    model_config = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    """Complete schema for one callable function.

    Loaded from the function definitions JSON file and used by the
    :class:`~src.decoder.candidate_builder.CandidateBuilder` to enumerate
    every valid JSON candidate for a given prompt.

    Attributes:
        name: Unique identifier for this function (used as the ``"name"``
            field in the decoded JSON output).
        description: Human-readable description used for LLM-driven function
            selection scoring.
        parameters: Ordered mapping of parameter name → definition.
        returns: Schema describing the function's return value.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition
