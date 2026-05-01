"""Output-facing Pydantic models for function-call results.

These models describe what the pipeline writes to disk.  They are separate
from the decoder-internal models (``FunctionDefinition`` etc.) so that
changes to the output schema do not accidentally affect decoding logic.
"""
from pydantic import BaseModel, ConfigDict


class DatasetSummary(BaseModel):
    """Lightweight statistics about a processed dataset.

    Useful for logging and debugging; not part of the primary output file.

    Attributes:
        function_count: Number of unique function definitions loaded.
        prompt_count: Number of prompts that were processed.
        average_prompt_length: Mean character length of all prompts.
    """

    model_config = ConfigDict(extra="forbid")

    function_count: int
    prompt_count: int
    average_prompt_length: float


class FunctionCallResult(BaseModel):
    """A single decoded function call paired with the prompt that produced it.

    This is the top-level object written to the output JSON file.  Each
    entry preserves the original prompt so that results can be traced back
    to their source without a separate lookup.

    Attributes:
        prompt: The original user-facing prompt text.
        name: The function name selected by the decoder.
        parameters: Mapping of parameter names to their decoded values.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]


class FunctionCallPayload(BaseModel):
    """The raw inner payload emitted by the decoder (no prompt context).

    The decoder produces JSON of the form::

        {"name": "...", "parameters": {...}}

    This model validates that JSON before it is wrapped into a
    :class:`FunctionCallResult`.

    Attributes:
        name: The function name chosen by the decoder.
        parameters: Mapping of parameter names to their decoded values.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: dict[str, object]
