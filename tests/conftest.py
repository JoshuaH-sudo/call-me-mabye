"""
Shared test fixtures and helpers.

CharLevelFakeLLM
----------------
A drop-in replacement for Small_LLM_Model that uses a single-character
vocabulary (token_id == ord(char) for ASCII 0-127).  It is constructed
with the plain-text prompt that will be encoded and the target JSON
string the decoder should produce.  At every generation step it returns
logits that give a score of 100.0 to the next character of the target
output, steering the constrained decoder to produce that exact string.
This lets us test decoder correctness without loading the real model.

FunctionDefinition fixtures
----------------------------
One pytest fixture per function from functions_definition.json so that
each test file can request exactly the functions it needs.
"""
import numpy as np
import pytest

from src.decoder_pipeline.models import (
    FunctionDefinition,
    ParameterDefinition,
    ReturnDefinition,
)


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class CharLevelFakeLLM:
    """
    Character-level mock LLM.

    Vocabulary
    ----------
    Token ID = ord(char) for printable ASCII (32-126).
    All other IDs decode to the empty string and are ignored by the
    decoder's token scanner.

    Parameters
    ----------
    prompt:
        The plain-text prompt that will be encoded before being passed to
        ``force_json_output``.  Used only to compute how many prompt
        tokens have been consumed so the mock can infer which step of
        ``target_output`` to prefer next.
    target_output:
        The JSON string the decoder should produce.  At step k the mock
        assigns logit 100.0 to ``ord(target_output[k])``, so the
        constrained decoder will always pick that character (as long as
        it is a valid continuation of the grammar).
    """

    VOCAB_SIZE: int = 128

    def __init__(self, prompt: str, target_output: str) -> None:
        # Count only tokens that survive the ASCII filter.
        self._prompt_token_len: int = sum(
            1 for c in prompt if ord(c) < self.VOCAB_SIZE
        )
        self._target: str = target_output

    def encode(self, text: str) -> np.ndarray:
        """Return a (1, N) int array of token IDs."""
        ids = [ord(c) for c in text if ord(c) < self.VOCAB_SIZE]
        return np.array([ids], dtype=np.int64)

    def decode(self, ids: "list[int] | np.ndarray") -> str:
        """Convert token IDs back to a string."""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return "".join(
            chr(i) for i in ids if 32 <= i < self.VOCAB_SIZE
        )

    def get_logits_from_input_ids(
        self, prefix_ids: "list[int]"
    ) -> "list[float]":
        """
        Return logits that prefer the next character of the target output.

        The rolling prefix length minus the prompt length gives the number
        of tokens generated so far, which is used as an index into
        ``target_output``.
        """
        logits = [0.0] * self.VOCAB_SIZE
        generated_len = len(prefix_ids) - self._prompt_token_len
        if 0 <= generated_len < len(self._target):
            char = self._target[generated_len]
            token_id = ord(char)
            if 0 <= token_id < self.VOCAB_SIZE:
                logits[token_id] = 100.0
        return logits


# ---------------------------------------------------------------------------
# Function-definition fixtures (mirror functions_definition.json)
# ---------------------------------------------------------------------------

@pytest.fixture
def fn_add_numbers() -> FunctionDefinition:
    """fn_add_numbers(a: number, b: number) -> number"""
    return FunctionDefinition(
        name="fn_add_numbers",
        description="Add two numbers together and return their sum.",
        parameters={
            "a": ParameterDefinition(type="number"),
            "b": ParameterDefinition(type="number"),
        },
        returns=ReturnDefinition(type="number"),
    )


@pytest.fixture
def fn_greet() -> FunctionDefinition:
    """fn_greet(name: string) -> string"""
    return FunctionDefinition(
        name="fn_greet",
        description=(
            "Generate a greeting message for a person by name."
        ),
        parameters={"name": ParameterDefinition(type="string")},
        returns=ReturnDefinition(type="string"),
    )


@pytest.fixture
def fn_reverse_string() -> FunctionDefinition:
    """fn_reverse_string(s: string) -> string"""
    return FunctionDefinition(
        name="fn_reverse_string",
        description="Reverse a string and return the reversed result.",
        parameters={"s": ParameterDefinition(type="string")},
        returns=ReturnDefinition(type="string"),
    )


@pytest.fixture
def fn_get_square_root() -> FunctionDefinition:
    """fn_get_square_root(a: number) -> number"""
    return FunctionDefinition(
        name="fn_get_square_root",
        description="Calculate the square root of a number.",
        parameters={"a": ParameterDefinition(type="number")},
        returns=ReturnDefinition(type="number"),
    )


@pytest.fixture
def fn_substitute_string_with_regex() -> FunctionDefinition:
    """fn_substitute_string_with_regex(source, regex, replacement)"""
    return FunctionDefinition(
        name="fn_substitute_string_with_regex",
        description=(
            "Replace all occurrences matching a regex pattern "
            "in a string."
        ),
        parameters={
            "source_string": ParameterDefinition(type="string"),
            "regex": ParameterDefinition(type="string"),
            "replacement": ParameterDefinition(type="string"),
        },
        returns=ReturnDefinition(type="string"),
    )


@pytest.fixture
def fn_unsupported_type() -> FunctionDefinition:
    """A function whose parameter type is not yet supported."""
    return FunctionDefinition(
        name="fn_check_flag",
        description="Check whether a flag is set.",
        parameters={"flag": ParameterDefinition(type="boolean")},
        returns=ReturnDefinition(type="boolean"),
    )
