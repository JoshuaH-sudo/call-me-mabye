"""Decoder sub-package — all token-level constrained decoding logic.

Public surface (without pulling in the LLM dependency at import time):

* :class:`~src.decoder.models.FunctionDefinition` — schema for one callable.
* :class:`~src.decoder.models.ParameterDefinition` — schema for one parameter.
* :class:`~src.decoder.models.ReturnDefinition` — schema for the return type.
* :class:`~src.decoder.candidate_builder.CandidateBuilder` — builds the JSON
  candidate strings that constrain decoding.
* :class:`~src.decoder.prefix_matcher.PrefixMatcher` — validates token
  sequences against precomputed candidates.
* :class:`~src.decoder.regex_token_validator.RegexTokenValidator` — stateless
  helper for checking regex prefix and completion validity.
* All type aliases from :mod:`src.decoder.types`.

The following classes are **not** exported here because they import
``llm_sdk`` (which depends on ``torch``).  Import them directly from their
submodules to avoid loading the LLM stack when only the schema models are
needed:

* :class:`~src.decoder.constrained_decoder.ConstrainedDecoder`
* :class:`~src.decoder.token_selector.TokenSelector`
* :class:`~src.decoder.regex_decoder.RegexConstrainedDecoder`
"""
from .models import FunctionDefinition, ParameterDefinition, ReturnDefinition
from .candidate_builder import CandidateBuilder
from .prefix_matcher import PrefixMatcher
from .regex_token_validator import RegexTokenValidator
from .types import (
    AllowedTokenIds,
    EncodedOutputCandidates,
    Logits,
    OutputCandidate,
    OutputCandidates,
    ParameterValue,
    ParameterValues,
    ParameterValueSpace,
    TokenIds,
)

__all__ = [
    "FunctionDefinition",
    "ParameterDefinition",
    "ReturnDefinition",
    "CandidateBuilder",
    "PrefixMatcher",
    "RegexTokenValidator",
    "ParameterValue",
    "ParameterValues",
    "ParameterValueSpace",
    "TokenIds",
    "EncodedOutputCandidates",
    "OutputCandidate",
    "OutputCandidates",
    "AllowedTokenIds",
    "Logits",
]
