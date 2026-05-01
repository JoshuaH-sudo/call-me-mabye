"""Unit tests for the enriched_prefix_ids path in ConstrainedDecoder."""
import json
from unittest.mock import MagicMock

import pytest

from src.decoder.constrained_decoder import ConstrainedDecoder
from src.decoder.models import FunctionDefinition, ParameterDefinition, ReturnDefinition


def _make_add_function() -> FunctionDefinition:
    return FunctionDefinition(
        name="add",
        description="Adds two numbers together.",
        parameters={
            "a": ParameterDefinition(type="number"),
            "b": ParameterDefinition(type="number"),
        },
        returns=ReturnDefinition(type="number"),
    )


def _token_ids_for(text: str) -> list[int]:
    """Deterministic fake tokeniser: each char maps to its ordinal."""
    return [ord(c) for c in text]


def _make_mock_llm(
    candidate_json: str,
) -> MagicMock:
    """Build a mock Small_LLM_Model that drives the decoder to *candidate_json*.

    The mock LLM:
    * ``encode(text)`` — wraps ``_token_ids_for(text)`` in a 2-D mock tensor.
    * ``decode(ids)`` — reconstructs the string from ordinals.
    * ``get_logits_from_input_ids`` — always returns a logit vector that
      assigns a very high score (100.0) to the correct next token and 0.0
      to everything else.  This guarantees the constrained decoder always
      picks the right token.
    """
    encoded_candidate = _token_ids_for(candidate_json)
    vocab_size = 256  # cover all ASCII ordinals

    def fake_encode(text: str) -> MagicMock:
        ids = _token_ids_for(text)
        tensor_mock = MagicMock()
        tensor_mock.__getitem__ = lambda self, idx: MagicMock(
            tolist=lambda: ids
        )
        return tensor_mock

    def fake_decode(ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)

    def fake_get_logits(prefix_ids: list[int]) -> list[float]:
        """Return high logit for the next correct candidate token."""
        already_generated = len(prefix_ids)
        logits = [0.0] * vocab_size
        # Determine how many tokens of the candidate have been emitted.
        # The rolling prefix starts with either the raw or enriched prompt
        # IDs; after those, the generated tokens appear.
        # We find the first position where the prefix diverges from the
        # candidate to know which candidate token to reinforce.
        if already_generated < len(encoded_candidate):
            next_token = encoded_candidate[already_generated]
            if 0 <= next_token < vocab_size:
                logits[next_token] = 100.0
        return logits

    llm = MagicMock()
    llm.encode.side_effect = fake_encode
    llm.decode.side_effect = fake_decode
    llm.get_logits_from_input_ids.side_effect = fake_get_logits
    return llm


class TestEnrichedPrefixIds:
    """Verify that apply_decoder uses enriched_prefix_ids when provided."""

    def _run_decoder(
        self,
        enriched_prefix_ids: list[int] | None,
        track_logit_calls: list[list[int]],
    ) -> str:
        fn = _make_add_function()
        candidate = '{"name":"add","parameters":{"a":2,"b":3}}'
        llm = _make_mock_llm(candidate)

        # Intercept logit calls to inspect the prefix being used.
        original_side_effect = llm.get_logits_from_input_ids.side_effect

        def recording_logits(prefix_ids: list[int]) -> list[float]:
            track_logit_calls.append(list(prefix_ids))
            return original_side_effect(prefix_ids)

        llm.get_logits_from_input_ids.side_effect = recording_logits

        decoder = ConstrainedDecoder(available_functions=[fn], llm=llm)
        prompt = "What is 2 plus 3?"
        prompt_ids = _token_ids_for(prompt)
        return decoder.apply_decoder(
            prefix_input_ids=prompt_ids,
            prompt=prompt,
            enriched_prefix_ids=enriched_prefix_ids,
        )

    def test_enriched_prefix_ids_used_as_initial_context(self) -> None:
        """When enriched_prefix_ids is provided, logit calls start from it."""
        enriched_ids = _token_ids_for(
            "<available_functions>add(a: float, b: float)</available_functions>"
            "<instruction>Select the correct function.</instruction>"
            "<question>What is 2 plus 3?</question>"
        )
        calls: list[list[int]] = []
        self._run_decoder(enriched_prefix_ids=enriched_ids, track_logit_calls=calls)

        assert calls, "expected at least one logit call"
        # The very first logit call must start with the enriched prefix.
        first_call_prefix = calls[0][: len(enriched_ids)]
        assert first_call_prefix == enriched_ids

    def test_raw_prompt_ids_used_when_no_enriched_prefix(self) -> None:
        """When enriched_prefix_ids is None, logit calls start from the raw prompt."""
        calls: list[list[int]] = []
        prompt = "What is 2 plus 3?"
        prompt_ids = _token_ids_for(prompt)
        self._run_decoder(enriched_prefix_ids=None, track_logit_calls=calls)

        assert calls, "expected at least one logit call"
        first_call_prefix = calls[0][: len(prompt_ids)]
        assert first_call_prefix == prompt_ids

    def test_enriched_prefix_does_not_affect_candidates(self) -> None:
        """Candidate building is driven by the raw prompt, not the enriched prefix."""
        fn = _make_add_function()
        candidate = '{"name":"add","parameters":{"a":2,"b":3}}'
        llm = _make_mock_llm(candidate)

        decoder = ConstrainedDecoder(available_functions=[fn], llm=llm)
        prompt = "What is 2 plus 3?"
        prompt_ids = _token_ids_for(prompt)
        enriched_ids = _token_ids_for(
            "<available_functions>add(a: float, b: float)</available_functions>"
            f"<question>{prompt}</question>"
        )

        result = decoder.apply_decoder(
            prefix_input_ids=prompt_ids,
            prompt=prompt,
            enriched_prefix_ids=enriched_ids,
        )
        parsed = json.loads(result)
        assert parsed["name"] == "add"
        assert "a" in parsed["parameters"]
        assert "b" in parsed["parameters"]

    def test_output_is_valid_json_with_enriched_prefix(self) -> None:
        calls: list[list[int]] = []
        enriched_ids = _token_ids_for("<question>What is 2 plus 3?</question>")
        result = self._run_decoder(
            enriched_prefix_ids=enriched_ids, track_logit_calls=calls
        )
        parsed = json.loads(result)
        assert parsed["name"] == "add"

    def test_output_is_valid_json_without_enriched_prefix(self) -> None:
        calls: list[list[int]] = []
        result = self._run_decoder(
            enriched_prefix_ids=None, track_logit_calls=calls
        )
        parsed = json.loads(result)
        assert parsed["name"] == "add"
