from typing import TypeAlias


ParameterValue: TypeAlias = object
ParameterValues: TypeAlias = dict[str, ParameterValue]
ParameterValueSpace: TypeAlias = dict[str, list[ParameterValue]]

TokenIds: TypeAlias = list[int]
EncodedOutputCandidates: TypeAlias = list[TokenIds]

OutputCandidate: TypeAlias = str
OutputCandidates: TypeAlias = list[OutputCandidate]

AllowedTokenIds: TypeAlias = list[int]
Logits: TypeAlias = list[float]
