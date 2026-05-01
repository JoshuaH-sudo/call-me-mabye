"""Type aliases used throughout the decoder sub-package.

Using named aliases instead of raw built-in types makes the signatures of
decoder functions self-documenting and easier to refactor if the underlying
representation changes.

Naming conventions
------------------
* ``*Ids``      — lists of integer token IDs.
* ``*Candidates`` — collections of JSON output strings or their token-ID
  encodings.
* ``*Value(s)`` — parameter values extracted from a prompt.
* ``Logits``    — raw model output scores (one float per vocabulary entry).
"""
from typing import TypeAlias

# --- Parameter value types ---------------------------------------------------

# A single extracted parameter value (any JSON-serialisable Python object).
ParameterValue: TypeAlias = object

# A complete set of parameter values for one function call candidate,
# keyed by parameter name.
ParameterValues: TypeAlias = dict[str, ParameterValue]

# The full space of candidate values for each parameter, before the
# cross-product expansion step.
ParameterValueSpace: TypeAlias = dict[str, list[ParameterValue]]

# --- Token ID types ----------------------------------------------------------

# A sequence of integer token IDs produced by the tokeniser.
TokenIds: TypeAlias = list[int]

# A list of pre-encoded output candidates (one TokenIds list per candidate).
EncodedOutputCandidates: TypeAlias = list[TokenIds]

# --- Output candidate types --------------------------------------------------

# A single JSON output candidate string, e.g.:
#   '{"name":"add","parameters":{"a":1,"b":2}}'
OutputCandidate: TypeAlias = str

# The full list of candidate strings for a given prompt.
OutputCandidates: TypeAlias = list[OutputCandidate]

# --- Token selection types ---------------------------------------------------

# The subset of vocabulary token IDs that are allowed at the current
# decoding step (i.e. they all continue at least one valid candidate).
AllowedTokenIds: TypeAlias = list[int]

# Raw model logits: one float per vocabulary entry.
Logits: TypeAlias = list[float]
