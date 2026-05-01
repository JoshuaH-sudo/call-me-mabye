"""Shared Pydantic models for function-call results and payloads.

``function_call`` — output-facing models that represent a decoded call and
    its surrounding context (the original prompt, selected function name,
    and resolved parameter values).

``validation``    — helpers that verify a raw decoded payload against the
    function schema loaded from disk.
"""
from .function_call import (
    DatasetSummary,
    FunctionCallPayload,
    FunctionCallResult,
)
from .validation import is_valid_parameter_value, validate_function_payload

__all__ = [
    "DatasetSummary",
    "FunctionCallPayload",
    "FunctionCallResult",
    "is_valid_parameter_value",
    "validate_function_payload",
]
